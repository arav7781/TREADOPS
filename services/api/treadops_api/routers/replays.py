from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from treadops_ml.replay import SCENARIOS

from .. import audit, jobs
from ..db import get_db
from ..errors import ApiError
from ..models import Inspection, Job
from ..security import Principal, require
from ..services.inspections import active_deployment

router = APIRouter(prefix="/v1", tags=["replay"])


class ReplayIn(BaseModel):
    scenario: str
    seed: int = Field(42, ge=0, le=2**31)
    rate: float = Field(2.0, gt=0, le=20)
    count: int = Field(600, ge=1, le=5000)


@router.get("/replays/scenarios")
def scenarios(p: Principal = Depends(require("viewer"))):
    from treadops_ml.common import load_yaml

    cfg = load_yaml("configs/replay.yaml")
    return {"items": [{"id": k, **v} for k, v in cfg["scenarios"].items()], "defaults":
            {"count": cfg["default_count"], "rate": cfg["default_rate_per_s"], "seed": cfg["seed"]},
            "disclosure": "Replays use real public photographs with SIMULATED operational metadata "
                          "(line, camera, timestamps). They are never training-eligible."}


@router.post("/replays", status_code=202)
def start(body: ReplayIn, request: Request, p: Principal = Depends(require("ml_engineer")),
          db: Session = Depends(get_db)):
    if body.scenario not in SCENARIOS:
        raise ApiError(422, "unknown_scenario", f"scenario must be one of {list(SCENARIOS)}")
    _ptr, dep = active_deployment(db)
    if dep is None:
        raise ApiError(503, "model_unavailable", "No trained model available; replays need an active deployment.")
    job = jobs.create_job(db, kind="replay", dedupe_key=f"replay:{uuid.uuid4()}", payload=body.model_dump(),
                          actor_id=p.id, correlation_id=request.state.request_id, resource_type="replay",
                          max_attempts=3)  # resumable: retries skip already-submitted events
    audit.record(db, actor=p, action="replay.requested", resource_type="job", resource_id=job.id,
                 correlation_id=request.state.request_id, payload=body.model_dump())
    db.commit()
    return {"job_id": str(job.id), "status": "queued", "metadata_origin": "simulated"}


@router.get("/replays")
def list_replays(p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    rows = db.execute(select(Job).where(Job.kind == "replay").order_by(Job.created_at.desc()).limit(20)).scalars().all()
    counts = dict(db.execute(select(Inspection.replay_job_id, func.count()).where(
        Inspection.replay_job_id.in_([j.id for j in rows])).group_by(Inspection.replay_job_id)).all()) if rows else {}
    done = dict(db.execute(select(Inspection.replay_job_id, func.count()).where(
        Inspection.replay_job_id.in_([j.id for j in rows]), Inspection.status == "succeeded")
        .group_by(Inspection.replay_job_id)).all()) if rows else {}
    return {"items": [{"job_id": str(j.id), "status": j.status, "progress": j.progress,
                       "progress_message": j.progress_message, "params": j.payload, "result": j.result,
                       "inspections": counts.get(j.id, 0), "completed": done.get(j.id, 0),
                       "created_at": j.created_at.isoformat(),
                       "error": j.error_message if j.status == "failed" else None} for j in rows]}
