"""Monitoring window: latest N completed inspections vs the active bundle's frozen reference cohort.

Opens (or merges into) a drift-review alert only after `consecutive_windows` alert-candidate windows on the
same feature. Alerts recommend investigation and label collection; they never retrain or promote.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy import and_, func, select

from treadops_api import audit, jobs
from treadops_api.db import new_session, session_scope
from treadops_api.models import DriftAlert, Inspection, Job, MonitoringWindow, Prediction, ReviewLabel
from treadops_api.services.inspections import active_deployment
from treadops_api.storage import get_storage
from treadops_ml.common import load_yaml, sha256_file, sha256_json
from treadops_ml.descriptors import DESCRIPTOR_NAMES
from treadops_ml.metrics import binary_metrics, latency_summary
from treadops_ml.monitoring import domain_classifier_auc, evidently_report, psi

from ..runtime import bundles
from ..tasks import JobContext, Permanent, handler


def _labelled(db, ids) -> dict:
    latest = (select(ReviewLabel.inspection_id, func.max(ReviewLabel.revision).label("rev"))
              .where(ReviewLabel.inspection_id.in_(ids)).group_by(ReviewLabel.inspection_id).subquery())
    return {r.inspection_id: r for r in db.execute(
        select(ReviewLabel).join(latest, and_(latest.c.inspection_id == ReviewLabel.inspection_id,
                                              latest.c.rev == ReviewLabel.revision))).scalars()}


@handler("monitoring")
def run(ctx: JobContext) -> dict:
    cfg = load_yaml("configs/drift.yaml")
    storage = get_storage()
    db = new_session()
    try:
        _ptr, dep = active_deployment(db)
        if dep is None:
            raise Permanent("model_unavailable", "no active deployment to monitor")
        bundle_dir = bundles().local_dir(dep.bundle_sha256, dep.model_version.bundle_uri)
        ref = pd.read_parquet(bundle_dir / "reference_features.parquet")
        pca = np.load(bundle_dir / "reference_pca.npz")
        reference_version = sha256_file(bundle_dir / "reference_features.parquet")[:16]
        rows = db.execute(
            select(Inspection, Prediction).join(Prediction, Prediction.inspection_id == Inspection.id)
            .where(Inspection.status == "succeeded").order_by(Inspection.completed_at.desc())
            .limit(int(cfg["window_size"]))
        ).all()
        if not rows:
            raise Permanent("no_data", "no completed inspections yet")
        ids = [i.id for i, _ in rows]
        parents = {str(i.image.parent_image_id or i.image_id) for i, _ in rows}
        cur = pd.DataFrame([{k: pr.quality["descriptors"][k] for k in DESCRIPTOR_NAMES} for _, pr in rows])
        meta = pd.DataFrame([{"camera_id": i.camera_id or "upload", "scenario_id": i.scenario_id or "none",
                              "triage": pr.triage, "p": pr.p_defective} for i, pr in rows])
        sufficient = len(parents) >= int(cfg["min_distinct_parents"])

        descriptor_psi = {k: psi(ref[k].to_numpy(), cur[k].to_numpy(), cfg["psi_bins"], cfg["psi_smoothing"])
                          for k in cfg["descriptors"]}
        ctx.heartbeat(0.3, "descriptor PSI done")

        emb = []
        for _insp, pr in rows:
            if pr.embedding_key:
                emb.append(np.load(io.BytesIO(storage.get(pr.embedding_key))))
        embedding = {"status": "insufficient_data"}
        if emb and cfg["embedding"]["enabled"]:
            proj = (np.stack(emb) - pca["mean"]) @ pca["components"].T
            ref_pc = ref[[c for c in ref.columns if c.startswith("pc_")]].to_numpy()
            embedding = domain_classifier_auc(ref_pc, proj, cfg["embedding"]["domain_classifier_seed"])
            embedding["pc_psi_mean"] = float(np.mean([psi(ref_pc[:, j], proj[:, j])["psi"]
                                                      for j in range(ref_pc.shape[1])]))
        ctx.heartbeat(0.55, "embedding drift done")

        gate_triage = (dep.model_version.metrics or {}).get("gate_cohort", {}).get("triage", {})
        ref_rate = {k: (gate_triage.get(k) or {}).get("fraction") for k in ("good_candidate", "manual_review",
                                                                           "defect_flagged")}
        cur_rate = meta.triage.value_counts(normalize=True).to_dict()
        prediction_shift = {"reference_triage_fraction": ref_rate, "current_triage_fraction": cur_rate,
                            "current_mean_p_defective": float(meta.p.mean()),
                            "note": "a prior shift is not automatically a model failure"}

        labels = _labelled(db, ids)
        lab_rows = [(pr, labels[i.id]) for i, pr in rows if i.id in labels and labels[i.id].label != "unjudgeable"]
        labelled = {"labelled": len(lab_rows), "coverage": len(lab_rows) / len(rows), "status": "insufficient_data"}
        if lab_rows:
            y = np.array([lb.label == "defective" for _, lb in lab_rows])
            pd_ = np.array([pr.p_defective for pr, _ in lab_rows])
            thr = json.loads((bundle_dir / "policy.json").read_text())["binary_threshold"]
            labelled.update(binary_metrics(pd_, y, float(thr)))
            ages = [(lb.verified_at - pr.created_at).total_seconds() / 3600 for pr, lb in lab_rows]
            labelled["median_label_age_h"] = float(np.median(ages))
            labelled["disagreement_rate"] = float(np.mean((pd_ >= float(thr)) != y))
            labelled["status"] = "ok" if len(lab_rows) >= 30 else "insufficient_data"

        q_ms = [pr.timing.get("queue_ms", 0) for _, pr in rows]
        tot_ms = [pr.timing.get("worker_total_ms", 0) for _, pr in rows]
        window_start = min(i.created_at for i, _ in rows)
        failed = db.execute(select(func.count()).select_from(Job).where(
            Job.kind == "inference", Job.status == "failed", Job.created_at >= window_start)).scalar_one()
        retries = db.execute(select(func.coalesce(func.sum(Job.attempt - 1), 0)).where(
            Job.kind == "inference", Job.created_at >= window_start, Job.attempt > 1)).scalar_one()
        system = {"queue_ms": latency_summary(q_ms), "worker_total_ms": latency_summary(tot_ms),
                  "failed_jobs": failed, "retries": int(retries)}

        breakdown = {"by_camera": meta.assign(brightness=cur.brightness).groupby("camera_id")
                     .agg(n=("p", "size"), mean_brightness=("brightness", "mean"), mean_p=("p", "mean"))
                     .round(4).reset_index().to_dict("records"),
                     "by_scenario": meta.assign(brightness=cur.brightness).groupby("scenario_id")
                     .agg(n=("p", "size"), mean_brightness=("brightness", "mean"), mean_p=("p", "mean"))
                     .round(4).reset_index().to_dict("records")}

        flagged = sorted(k for k, v in descriptor_psi.items() if v["psi"] > cfg["psi_alert_candidate"])
        if embedding.get("auc") and embedding["auc"] > cfg["embedding"]["auc_alert"]:
            flagged.append("embedding")
        status = "insufficient_data" if not sufficient else ("alert_candidate" if flagged else "ok")

        ref_df = ref[list(cfg["descriptors"])]
        ev = evidently_report(ref_df, cur, list(cfg["descriptors"]))
        report_key = None
        if ev.get("status") == "ok":
            report_key = f"monitoring/{ctx.job_id}/evidently.json"
            storage.put(report_key, json.dumps(ev.pop("full"), default=str).encode(), "application/json")
        cohort = sha256_json(sorted(str(i) for i in ids))
        db.close()
    except Exception:
        db.close()
        raise

    with session_scope() as s:
        window = MonitoringWindow(
            window_key=f"{dep.id}:{ids[0]}", deployment_id=dep.id, reference_version=reference_version,
            cohort_sha256=cohort, n_inspections=len(rows), n_distinct_parents=len(parents),
            first_at=window_start, last_at=max(i.completed_at for i, _ in rows), status=status,
            descriptor_psi=descriptor_psi, prediction_shift=prediction_shift, embedding_drift=embedding,
            labelled_quality=labelled, system_health=system, breakdown=breakdown, report_key=report_key,
            evidently=ev,
        )
        existing = s.execute(select(MonitoringWindow).where(MonitoringWindow.window_key == window.window_key)
                             ).scalar_one_or_none()
        if existing is None:
            s.add(window)
            s.flush()
        else:
            window = existing
        alerts = []
        if status == "alert_candidate":
            prev = s.execute(select(MonitoringWindow).where(MonitoringWindow.id != window.id,
                                                            MonitoringWindow.created_at < window.created_at)
                             .order_by(MonitoringWindow.created_at.desc()).limit(int(cfg["consecutive_windows"]) - 1)
                             ).scalars().all()
            for feature in flagged:
                persistent = all(w.status == "alert_candidate" and (
                    (feature == "embedding" and (w.embedding_drift or {}).get("auc", 0) > cfg["embedding"]["auc_alert"])
                    or (w.descriptor_psi.get(feature, {}).get("psi", 0) > cfg["psi_alert_candidate"]))
                    for w in prev) and len(prev) == int(cfg["consecutive_windows"]) - 1
                if not persistent:
                    continue
                key = f"{feature}:{reference_version}"
                alert = s.execute(select(DriftAlert).where(DriftAlert.alert_key == key,
                                                           DriftAlert.state != "resolved")).scalar_one_or_none()
                value = embedding.get("auc") if feature == "embedding" else descriptor_psi[feature]["psi"]
                top_cams = sorted(breakdown["by_camera"], key=lambda r: r["mean_brightness"])[:3]
                details = {"feature": feature, "value": value, "threshold": cfg["psi_alert_candidate"]
                           if feature != "embedding" else cfg["embedding"]["auc_alert"],
                           "affected": {"cameras": top_cams, "scenarios": breakdown["by_scenario"]},
                           "sample_inspection_ids": [str(i) for i in ids[:6]], "labelled_quality": labelled}
                if alert is None:
                    alert = DriftAlert(alert_key=key, state="open", severity="high" if (value or 0) > 0.5 else "medium",
                                       features=[feature], reference_version=reference_version, cohort_sha256=cohort,
                                       window_ids=[str(w.id) for w in prev] + [str(window.id)], details=details,
                                       recommendation=("Inspect affected cameras/scenarios and sample images; collect "
                                                       "verified labels to measure quality. Retraining and promotion "
                                                       "remain separate gated operations."))
                    s.add(alert)
                    audit.record(s, action="drift.alert.opened", resource_type="drift_alert", resource_id=alert.alert_key,
                                 payload={"feature": feature, "value": value})
                else:  # merge duplicate alerts into the open one
                    alert.window_ids = [*alert.window_ids, str(window.id)]
                    alert.details = details
                    alert.updated_at = datetime.now(UTC)
                alerts.append(key)
        result = {"window_id": str(window.id), "status": status, "n": len(rows), "distinct_parents": len(parents),
                  "flagged": flagged, "alerts": alerts, "evidently": ev.get("status")}
        jobs.succeed(s, s.get(Job, ctx.job_id, populate_existing=True), result)
    return result
