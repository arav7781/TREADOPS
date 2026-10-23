from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import audit, jobs
from ..db import get_db
from ..errors import ApiError
from ..models import Deployment, Job, ModelVersion, ShadowResult
from ..security import Principal, require
from ..services.inspections import active_deployment
from ..services.releases import model_view
from ..storage import get_storage

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
def list_models(p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    ptr, dep = active_deployment(db)
    db.commit()
    versions = db.execute(select(ModelVersion).order_by(ModelVersion.created_at.desc())).scalars().all()
    deps = db.execute(select(Deployment).order_by(Deployment.generation)).scalars().all()
    by_mv: dict = {}
    for d in deps:
        by_mv.setdefault(d.model_version_id, []).append(
            {"generation": d.generation, "action": d.action, "state": d.state, "at": d.created_at.isoformat()})
    active_id = dep.model_version_id if dep else None
    challenger = next((v for v in versions if v.gate_status == "passed" and v.id != active_id
                       and v.stage in ("validated", "shadow", "approved")), None)
    return {
        "active": model_view(dep.model_version, active_id) if dep else None,
        "challenger": model_view(challenger, active_id) if challenger else None,
        "generation": ptr.generation,
        "items": [model_view(v, active_id, by_mv.get(v.id)) for v in versions],
        "state": "ready" if dep else "untrained",
        "message": None if dep else "No trained model available",
    }


def _bundle_text(mv: ModelVersion, rel: str) -> str | None:
    try:
        return get_storage().get(f"bundles/{mv.bundle_sha256}/{rel}").decode()
    except FileNotFoundError:
        return None


@router.get("/models/{model_version_id}/evidence")
def evidence(model_version_id: uuid.UUID, p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    mv = db.get(ModelVersion, model_version_id)
    if mv is None:
        raise ApiError(404, "not_found", "Model version not found.")
    m = mv.metrics or {}
    cal = _bundle_text(mv, "calibration.json")
    pol = _bundle_text(mv, "policy.json")
    shadow_job = db.execute(select(Job).where(Job.kind == "shadow", Job.resource_id == str(mv.id))
                            .order_by(Job.created_at.desc()).limit(1)).scalar_one_or_none()
    shadow = None
    if shadow_job is not None:
        n = db.execute(select(func.count()).select_from(ShadowResult)
                       .where(ShadowResult.job_id == shadow_job.id)).scalar_one()
        shadow = {"job_id": str(shadow_job.id), "status": shadow_job.status, "rows": n, "report": shadow_job.result}
    return {
        "model": model_view(mv),
        "model_card": _bundle_text(mv, "model_card.md"),
        "gate_report": mv.gate_report,
        "gate_cohort": m.get("gate_cohort"),
        "locked_test": m.get("locked_test"),
        "stress": m.get("stress"),
        "parity": m.get("parity"),
        "latency": m.get("latency"),
        "comparative": m.get("comparative"),
        "training": m.get("training"),
        "calibration": json.loads(cal) if cal else None,
        "policy": json.loads(pol) if pol else None,
        "manifest": {k: mv.manifest.get(k) for k in ("reproducibility", "training", "evaluation", "serving",
                                                      "monitoring", "lineage", "bundle_sha256")},
        "shadow": shadow,
    }


class ShadowIn(BaseModel):
    max_images: int = Field(default=120, ge=20, le=400)
    scenarios: list[str] = Field(default_factory=lambda: ["baseline", "low_light", "camera_blur"])


@router.post("/models/{model_version_id}/shadow", status_code=202)
def start_shadow(model_version_id: uuid.UUID, request: Request, body: ShadowIn | None = None,
                 p: Principal = Depends(require("ml_engineer")), db: Session = Depends(get_db)):
    body = body or ShadowIn()
    mv = db.get(ModelVersion, model_version_id)
    if mv is None:
        raise ApiError(404, "not_found", "Model version not found.")
    _ptr, dep = active_deployment(db)
    if dep is None:
        raise ApiError(409, "no_champion", "Shadow evaluation needs an active champion to compare against.")
    if dep.model_version_id == mv.id:
        raise ApiError(409, "already_active", "The active model cannot shadow itself.")
    job = jobs.create_job(db, kind="shadow", dedupe_key=f"shadow:{mv.id}:{uuid.uuid4().hex[:8]}",
                          payload={"challenger_id": str(mv.id), "champion_id": str(dep.model_version_id),
                                   "max_images": body.max_images, "scenarios": body.scenarios},
                          actor_id=p.id, correlation_id=request.state.request_id, resource_type="model_version",
                          resource_id=str(mv.id))
    audit.record(db, actor=p, action="model.shadow.requested", resource_type="model_version", resource_id=mv.id,
                 correlation_id=request.state.request_id, payload={"job_id": str(job.id), **body.model_dump()})
    db.commit()
    return {"job_id": str(job.id), "status": "queued"}
