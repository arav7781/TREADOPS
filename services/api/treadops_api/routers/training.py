from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from treadops_ml.common import load_yaml, repo_path, sha256_json

from .. import audit, jobs
from ..db import get_db
from ..errors import ApiError
from ..models import DatasetSnapshot, Job, ModelVersion, TrainingRun
from ..security import Principal, require

router = APIRouter(prefix="/v1", tags=["training"])


def recipes() -> dict[str, dict]:
    """Allowlist = the files in configs/recipes. The API never accepts paths or commands."""
    return {p.stem: load_yaml(p) for p in sorted(repo_path("configs/recipes").glob("*.yaml"))}


@router.get("/training/recipes")
def list_recipes(p: Principal = Depends(require("viewer"))):
    return {"items": [{"id": k, **v, "sha256": sha256_json(v)} for k, v in recipes().items()]}


class TrainIn(BaseModel):
    snapshot_id: uuid.UUID
    config_id: str = Field(max_length=40)


def run_view(r: TrainingRun, job: Job | None = None) -> dict:
    return {
        "id": str(r.id), "snapshot_id": str(r.snapshot_id), "config_id": r.config_id, "config_sha256": r.config_sha256,
        "recipe": r.recipe, "status": r.status, "mlflow_run_id": r.mlflow_run_id, "epoch_metrics": r.epoch_metrics,
        "metrics": r.metrics, "artifact_uri": r.artifact_uri, "checkpoint_sha256": r.checkpoint_sha256,
        "model_version_id": str(r.model_version_id) if r.model_version_id else None, "error": r.error,
        "created_at": r.created_at.isoformat(), "started_at": r.started_at.isoformat() if r.started_at else None,
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        "job": {"job_id": str(job.id), "status": job.status, "progress": job.progress,
                "progress_message": job.progress_message, "attempt": job.attempt} if job else None,
    }


@router.post("/training-runs", status_code=202)
def start(body: TrainIn, request: Request, p: Principal = Depends(require("ml_engineer")),
          db: Session = Depends(get_db)):
    allow = recipes()
    if body.config_id not in allow:
        raise ApiError(422, "unknown_recipe", f"config_id must be one of {sorted(allow)}")
    snap = db.get(DatasetSnapshot, body.snapshot_id)
    if snap is None:
        raise ApiError(404, "not_found", "Snapshot not found.")
    active = db.execute(select(Job).where(Job.kind == "training", Job.status.in_(
        ("queued", "running", "retry_wait", "cancellation_requested"))).with_for_update()).scalars().first()
    if active is not None:
        raise ApiError(409, "training_slot_busy", "A training run is already active (single GPU/training slot).",
                       {"job_id": str(active.id)})
    run = TrainingRun(snapshot_id=snap.id, config_id=body.config_id, config_sha256=sha256_json(allow[body.config_id]),
                      recipe=allow[body.config_id], status="queued", created_by=p.id)
    db.add(run)
    db.flush()
    job = jobs.create_job(db, kind="training", dedupe_key=f"training:{run.id}", payload={"training_run_id": str(run.id)},
                          actor_id=p.id, correlation_id=request.state.request_id, resource_type="training_run",
                          resource_id=str(run.id), max_attempts=1)
    run.job_id = job.id
    audit.record(db, actor=p, action="training.requested", resource_type="training_run", resource_id=run.id,
                 correlation_id=request.state.request_id,
                 payload={"snapshot_id": str(snap.id), "config_id": body.config_id})
    db.commit()
    return {"training_run_id": str(run.id), "job_id": str(job.id), "status": "queued"}


@router.get("/training-runs")
def list_runs(p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    rows = db.execute(select(TrainingRun, Job).outerjoin(Job, Job.id == TrainingRun.job_id)
                      .order_by(TrainingRun.created_at.desc()).limit(50)).all()
    return {"items": [run_view(r, j) for r, j in rows]}


@router.get("/training-runs/{run_id}")
def get_run(run_id: uuid.UUID, p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    r = db.get(TrainingRun, run_id)
    if r is None:
        raise ApiError(404, "not_found", "Training run not found.")
    out = run_view(r, db.get(Job, r.job_id) if r.job_id else None)
    if r.model_version_id:
        mv = db.get(ModelVersion, r.model_version_id)
        out["model_version"] = {"id": str(mv.id), "gate_status": mv.gate_status, "stage": mv.stage,
                                "registry_version": mv.registry_version, "bundle_sha256": mv.bundle_sha256}
    return out
