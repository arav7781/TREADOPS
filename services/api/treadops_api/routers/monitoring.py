from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from treadops_ml.common import load_yaml

from .. import audit, jobs
from ..db import get_db
from ..errors import ApiError
from ..models import DriftAlert, DriftAlertReview, MonitoringWindow, User
from ..security import Principal, require
from ..storage import get_storage

router = APIRouter(prefix="/v1", tags=["monitoring"])
ALERT_FLOW = {"open": {"acknowledged", "investigating", "resolved"}, "acknowledged": {"investigating", "resolved"},
              "investigating": {"resolved"}, "resolved": set()}


def window_view(w: MonitoringWindow) -> dict:
    return {"id": str(w.id), "status": w.status, "created_at": w.created_at.isoformat(),
            "n_inspections": w.n_inspections, "n_distinct_parents": w.n_distinct_parents,
            "first_at": w.first_at.isoformat() if w.first_at else None,
            "last_at": w.last_at.isoformat() if w.last_at else None,
            "reference_version": w.reference_version, "cohort_sha256": w.cohort_sha256,
            "descriptor_psi": w.descriptor_psi, "prediction_shift": w.prediction_shift,
            "embedding_drift": w.embedding_drift, "labelled_quality": w.labelled_quality,
            "system_health": w.system_health, "breakdown": w.breakdown, "evidently": w.evidently,
            "has_report": bool(w.report_key)}


@router.get("/monitoring/windows")
def windows(limit: int = Query(20, ge=1, le=100), p: Principal = Depends(require("viewer")),
            db: Session = Depends(get_db)):
    rows = db.execute(select(MonitoringWindow).order_by(MonitoringWindow.created_at.desc()).limit(limit)).scalars().all()
    latest = rows[0] if rows else None
    cfg = load_yaml("configs/drift.yaml")
    return {"items": [window_view(w) for w in rows],
            "thresholds": {"psi_alert_candidate": cfg["psi_alert_candidate"], "consecutive_windows":
                           cfg["consecutive_windows"], "min_distinct_parents": cfg["min_distinct_parents"],
                           "window_size": cfg["window_size"], "evaluate_every": cfg["evaluate_every"],
                           "embedding_auc_alert": cfg["embedding"]["auc_alert"]},
            "state": "no_windows" if latest is None else latest.status,
            "message": None if rows else "No monitoring window yet: one is computed every 200 completed inspections "
                                         "(or run one now)."}


@router.post("/monitoring/windows", status_code=202)
def run_now(request: Request, p: Principal = Depends(require("ml_engineer")), db: Session = Depends(get_db)):
    job = jobs.create_job(db, kind="monitoring", dedupe_key=f"monitoring:manual:{uuid.uuid4()}",
                          payload={"trigger": "manual"}, actor_id=p.id, correlation_id=request.state.request_id)
    db.commit()
    return {"job_id": str(job.id), "status": "queued"}


@router.get("/monitoring/windows/{window_id}/report")
def report(window_id: uuid.UUID, p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    w = db.get(MonitoringWindow, window_id)
    if w is None or not w.report_key:
        raise ApiError(404, "not_found", "No Evidently report stored for this window.")
    import json

    return json.loads(get_storage().get(w.report_key))


@router.get("/drift-alerts")
def alerts(state: str | None = None, p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    q = select(DriftAlert)
    if state:
        q = q.where(DriftAlert.state == state)
    rows = db.execute(q.order_by(DriftAlert.created_at.desc()).limit(100)).scalars().all()
    out = []
    for a in rows:
        reviews = db.execute(select(DriftAlertReview, User).join(User, User.id == DriftAlertReview.actor_id)
                             .where(DriftAlertReview.alert_id == a.id).order_by(DriftAlertReview.id)).all()
        out.append({"id": str(a.id), "state": a.state, "severity": a.severity, "features": a.features,
                    "reference_version": a.reference_version, "cohort_sha256": a.cohort_sha256,
                    "window_ids": a.window_ids, "details": a.details, "recommendation": a.recommendation,
                    "created_at": a.created_at.isoformat(), "updated_at": a.updated_at.isoformat(),
                    "reviews": [{"from": r.from_state, "to": r.to_state, "reason": r.reason, "by": u.display_name,
                                 "at": r.at.isoformat()} for r, u in reviews]})
    return {"items": out}


class AlertReviewIn(BaseModel):
    state: str
    reason: str = Field(min_length=3, max_length=2000)


@router.post("/drift-alerts/{alert_id}/review")
def review_alert(alert_id: uuid.UUID, body: AlertReviewIn, request: Request,
                 p: Principal = Depends(require("ml_engineer")), db: Session = Depends(get_db)):
    a = db.execute(select(DriftAlert).where(DriftAlert.id == alert_id).with_for_update()).scalar_one_or_none()
    if a is None:
        raise ApiError(404, "not_found", "Alert not found.")
    if body.state not in ALERT_FLOW.get(a.state, set()):
        raise ApiError(409, "invalid_transition", f"Cannot move an alert from {a.state} to {body.state}.")
    db.add(DriftAlertReview(alert_id=a.id, from_state=a.state, to_state=body.state, reason=body.reason,
                            actor_id=p.id))
    audit.record(db, actor=p, action="drift.alert.review", resource_type="drift_alert", resource_id=a.id,
                 correlation_id=request.state.request_id, payload={"from": a.state, "to": body.state,
                                                                   "reason": body.reason})
    a.state = body.state
    db.commit()
    return {"id": str(a.id), "state": a.state, "note": "Reviewing an alert never promotes or retrains a model."}
