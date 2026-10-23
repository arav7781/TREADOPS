from __future__ import annotations

import uuid
from collections.abc import Sequence

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import audit
from .. import jobs as jobsvc
from ..db import get_db
from ..errors import ApiError
from ..models import Job, JobEvent
from ..security import Principal, current_principal, require

router = APIRouter(prefix="/v1", tags=["jobs"])
ML_KINDS = {"training", "snapshot", "replay", "shadow", "monitoring", "registry_reconcile"}


def job_view(job: Job, events: Sequence[JobEvent] | None = None) -> dict:
    out = {
        "job_id": str(job.id), "kind": job.kind, "queue": job.queue, "status": job.status, "attempt": job.attempt,
        "max_attempts": job.max_attempts, "progress": job.progress, "progress_message": job.progress_message,
        "error": {"code": job.error_code, "message": job.error_message, "retryable": job.retryable}
        if job.error_code else None,
        "cancel_requested": job.cancel_requested, "result": job.result,
        "resource": {"type": job.resource_type, "id": job.resource_id},
        "created_at": job.created_at.isoformat(), "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "next_retry_at": job.next_retry_at.isoformat() if job.next_retry_at else None,
        "correlation_id": job.correlation_id,
    }
    if events is not None:
        out["events"] = [{"at": e.at.isoformat(), "from": e.from_status, "to": e.to_status, "attempt": e.attempt,
                          "reason": e.reason} for e in events]
    return out


def _authorize(p: Principal, job: Job) -> None:
    if job.created_by == p.id or p.has("admin"):
        return
    if job.kind in ML_KINDS and p.has("ml_engineer"):
        return
    if job.kind == "inference" and p.has("viewer"):
        return
    raise ApiError(403, "forbidden", "Only the job owner or an admin can view this job.")


@router.get("/jobs/{job_id}")
def get_job(job_id: uuid.UUID, p: Principal = Depends(current_principal), db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise ApiError(404, "not_found", "Job not found.")
    _authorize(p, job)
    events = db.execute(select(JobEvent).where(JobEvent.job_id == job_id).order_by(JobEvent.id)).scalars().all()
    return job_view(job, events)


class CancelIn(BaseModel):
    reason: str = Field("cancelled by user", max_length=500)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: uuid.UUID, request: Request, body: CancelIn | None = None,
               p: Principal = Depends(current_principal), db: Session = Depends(get_db)):
    job = db.execute(select(Job).where(Job.id == job_id).with_for_update()).scalar_one_or_none()
    if job is None:
        raise ApiError(404, "not_found", "Job not found.")
    if not (job.created_by == p.id or p.has("admin")):
        raise ApiError(403, "forbidden", "Only the job owner or an admin can cancel this job.")
    before = job.status
    jobsvc.request_cancel(db, job, (body.reason if body else "cancelled by user"))
    if job.status != before:
        audit.record(db, actor=p, action="job.cancel", resource_type="job", resource_id=job.id,
                     correlation_id=request.state.request_id, payload={"from": before, "to": job.status})
    db.commit()
    return job_view(job)


@router.get("/jobs")
def list_jobs(kind: str | None = None, status: str | None = None, limit: int = Query(50, ge=1, le=100),
              p: Principal = Depends(require("ml_engineer")), db: Session = Depends(get_db)):
    q = select(Job)
    if kind:
        q = q.where(Job.kind == kind)
    if status:
        q = q.where(Job.status == status)
    rows = db.execute(q.order_by(Job.created_at.desc()).limit(limit)).scalars().all()
    counts = dict(db.execute(select(Job.status, func.count()).group_by(Job.status)).all())
    return {"items": [job_view(j) for j in rows], "counts_by_status": counts}
