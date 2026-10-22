"""Dashboard aggregates, computed from stored inspections and jobs at request time. Nothing is hard-coded."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Float, and_, cast, func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import DriftAlert, Inspection, Job, MonitoringWindow, Prediction, ReviewLabel, WorkerHeartbeat
from ..security import Principal, require
from ..services.inspections import active_deployment

router = APIRouter(prefix="/v1", tags=["metrics"])


def _pct(col, q):
    return func.percentile_cont(q).within_group(col)


@router.get("/metrics/overview")
def overview(
    since: datetime | None = None,
    until: datetime | None = None,
    source: str | None = None,
    line_id: str | None = None,
    camera_id: str | None = None,
    metadata_origin: str | None = None,
    bucket: str = Query("hour", pattern="^(minute|hour|day)$"),
    p: Principal = Depends(require("viewer")),
    db: Session = Depends(get_db),
):
    now = datetime.now(UTC)
    since = since or now - timedelta(days=7)
    until = until or now
    conds = [Inspection.created_at >= since, Inspection.created_at < until]
    for col, val in ((Inspection.source, source), (Inspection.line_id, line_id), (Inspection.camera_id, camera_id),
                     (Inspection.metadata_origin, metadata_origin)):
        if val:
            conds.append(col == val)
    where = and_(*conds)

    by_status = dict(db.execute(select(Inspection.status, func.count()).where(where)
                                .group_by(Inspection.status)).all())
    joined = select(Prediction).join(Inspection, Inspection.id == Prediction.inspection_id).where(where).subquery()
    by_triage = dict(db.execute(select(joined.c.triage, func.count()).group_by(joined.c.triage)).all())
    by_quality = dict(db.execute(select(joined.c.quality_status, func.count()).group_by(joined.c.quality_status)).all())
    by_origin = dict(db.execute(select(Inspection.metadata_origin, func.count()).where(where)
                                .group_by(Inspection.metadata_origin)).all())
    lat = db.execute(select(
        _pct(cast(joined.c.timing["worker_total_ms"].astext, Float), 0.5),
        _pct(cast(joined.c.timing["worker_total_ms"].astext, Float), 0.95),
        _pct(cast(joined.c.timing["inference_ms"].astext, Float), 0.5),
        _pct(cast(joined.c.timing["inference_ms"].astext, Float), 0.95),
        _pct(cast(joined.c.timing["queue_ms"].astext, Float), 0.95),
        func.count(),
    )).one()
    trunc = func.date_trunc(bucket, Inspection.created_at)
    series = db.execute(
        select(trunc.label("t"), Prediction.triage, func.count(),
               func.avg(cast(Prediction.timing["worker_total_ms"].astext, Float)))
        .join(Prediction, Prediction.inspection_id == Inspection.id).where(where)
        .group_by("t", Prediction.triage).order_by("t")
    ).all()
    timeline: dict[str, dict] = {}
    for t, triage, n, avg_ms in series:
        row = timeline.setdefault(t.isoformat(), {"t": t.isoformat(), "good_candidate": 0, "manual_review": 0,
                                                  "defect_flagged": 0, "avg_total_ms": None})
        row[triage] = n
        row["avg_total_ms"] = round(float(avg_ms), 1) if avg_ms is not None else row["avg_total_ms"]

    has_review = select(ReviewLabel.id).where(ReviewLabel.inspection_id == Inspection.id).exists()
    review_q = select(func.count(), func.min(Inspection.created_at)).select_from(Inspection).where(
        Inspection.status == "succeeded", ~has_review)
    awaiting, oldest_review = db.execute(review_q).one()
    queue = db.execute(select(Job.status, func.count(), func.min(Job.created_at)).where(
        Job.kind == "inference", Job.status.in_(("queued", "running", "retry_wait"))).group_by(Job.status)).all()
    oldest_job = min((r[2] for r in queue if r[0] == "queued"), default=None)
    fresh = now - timedelta(seconds=45)
    workers = db.execute(select(WorkerHeartbeat)).scalars().all()
    ptr, dep = active_deployment(db)
    db.commit()
    window = db.execute(select(MonitoringWindow).order_by(MonitoringWindow.created_at.desc()).limit(1)).scalar_one_or_none()
    open_alerts = db.execute(select(func.count()).select_from(DriftAlert).where(DriftAlert.state != "resolved")).scalar_one()
    return {
        "observed_at": now.isoformat(),
        "filters": {"since": since.isoformat(), "until": until.isoformat(), "source": source, "line_id": line_id,
                    "camera_id": camera_id, "metadata_origin": metadata_origin, "bucket": bucket},
        "cohort_size": sum(by_status.values()),
        "inspections": {"by_status": by_status, "by_triage": by_triage, "by_quality": by_quality,
                        "by_metadata_origin": by_origin},
        "latency_ms": {"n": lat[5], "worker_total_p50": lat[0], "worker_total_p95": lat[1], "inference_p50": lat[2],
                       "inference_p95": lat[3], "queue_p95": lat[4]},
        "timeline": list(timeline.values()),
        "review": {"awaiting": awaiting, "oldest_waiting_at": oldest_review.isoformat() if oldest_review else None},
        "queue": {"by_status": {r[0]: r[1] for r in queue},
                  "oldest_queued_age_s": (now - oldest_job).total_seconds() if oldest_job else None},
        "workers": [{"name": w.worker_name, "queue": w.queue, "alive": w.last_seen_at > fresh, "paused": w.paused,
                     "last_seen_at": w.last_seen_at.isoformat(), "ready_bundles": w.ready_bundles} for w in workers],
        "active_model": {"state": "ready" if dep else "untrained", "generation": ptr.generation,
                         "registry_version": dep.model_version.registry_version if dep else None,
                         "bundle_sha256": dep.bundle_sha256 if dep else None,
                         "policy_version": dep.policy_version if dep else None,
                         "gate_status": dep.model_version.gate_status if dep else None,
                         "snapshot": (dep.model_version.manifest or {}).get("lineage", {}).get("dataset_snapshot_id")
                         if dep else None,
                         "message": None if dep else "No trained model available"},
        "drift": {"latest_window": {"status": window.status, "created_at": window.created_at.isoformat(),
                                    "n": window.n_inspections, "distinct_parents": window.n_distinct_parents}
                  if window else None,
                  "open_alerts": open_alerts},
    }
