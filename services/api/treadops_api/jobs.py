"""Durable job state machine + transactional outbox + leases.

    queued -> running -> succeeded | failed | retry_wait -> queued
    queued/retry_wait -> cancelled ; running -> cancellation_requested -> cancelled
Terminal states are immutable. Every transition writes a JobEvent with attempt, reason and correlation id.
Celery delivery is at-least-once, so `claim` is a compare-and-set: a redelivered message for a job that is
already running (live lease) or finished is a no-op.
"""

from __future__ import annotations

import random
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, text, update
from sqlalchemy.orm import Session

from .models import Job, JobEvent, OutboxEvent

KIND_QUEUE = {
    "inference": "inference",
    "training": "training",
    "snapshot": "ops",
    "replay": "ops",
    "monitoring": "ops",
    "promotion": "ops",
    "rollback": "ops",
    "shadow": "ops",
    "registry_reconcile": "ops",
}
TERMINAL = {"succeeded", "failed", "cancelled"}
ALLOWED = {
    "queued": {"running", "cancelled"},
    "running": {"succeeded", "failed", "retry_wait", "cancellation_requested"},
    "retry_wait": {"queued", "cancelled", "failed"},
    "cancellation_requested": {"cancelled", "succeeded", "failed", "retry_wait"},
}
BACKOFF_S = (5, 15, 45)


class InvalidTransition(RuntimeError):
    pass


def now() -> datetime:
    return datetime.now(UTC)


def _event(db: Session, job: Job, frm: str | None, to: str, reason: str | None = None) -> None:
    db.add(JobEvent(job_id=job.id, from_status=frm, to_status=to, attempt=job.attempt, reason=reason,
                    correlation_id=job.correlation_id))


def enqueue_outbox(db: Session, job: Job, delay_s: float = 0.0) -> OutboxEvent:
    ev = OutboxEvent(topic="job.dispatch", payload={"job_id": str(job.id), "kind": job.kind, "queue": job.queue,
                                                    "attempt": job.attempt},
                     available_at=now() + timedelta(seconds=delay_s))
    db.add(ev)
    return ev


def create_job(
    db: Session,
    *,
    kind: str,
    dedupe_key: str,
    payload: dict[str, Any],
    actor_id: uuid.UUID | None = None,
    correlation_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    max_attempts: int = 3,
    job_id: uuid.UUID | None = None,
) -> Job:
    """Insert job + first event + outbox row in the caller's transaction (commit makes all three durable)."""
    job = Job(
        id=job_id or uuid.uuid4(), kind=kind, queue=KIND_QUEUE[kind], dedupe_key=dedupe_key, status="queued",
        attempt=0, max_attempts=max_attempts, payload=payload, created_by=actor_id,
        correlation_id=correlation_id or uuid.uuid4().hex, resource_type=resource_type, resource_id=resource_id,
    )
    db.add(job)
    db.flush()
    _event(db, job, None, "queued", "accepted")
    enqueue_outbox(db, job)
    return job


def transition(db: Session, job: Job, to: str, reason: str | None = None) -> None:
    frm = job.status
    if frm in TERMINAL:
        raise InvalidTransition(f"job {job.id} is terminal ({frm})")
    if to not in ALLOWED.get(frm, set()):
        raise InvalidTransition(f"{frm} -> {to} not allowed")
    job.status = to
    job.updated_at = now()
    if to in TERMINAL:
        job.finished_at = now()
        job.lease_owner = None
        job.lease_expires_at = None
    _event(db, job, frm, to, reason)


def claim(db: Session, job_id: uuid.UUID, owner: str, lease_s: int) -> Job | None:
    """CAS queued -> running with a lease. Returns the job when this worker owns it, else None."""
    res = db.execute(
        update(Job)
        .where(Job.id == job_id, Job.status == "queued")
        .values(status="running", attempt=Job.attempt + 1, lease_owner=owner,
                lease_expires_at=now() + timedelta(seconds=lease_s), started_at=now(), updated_at=now(),
                next_retry_at=None)
        .returning(Job.id)
    ).first()
    if res is None:
        db.rollback()
        return None
    job = db.get(Job, job_id, populate_existing=True)
    _event(db, job, "queued", "running", f"claimed by {owner}")
    db.commit()
    return job


def heartbeat(db: Session, job_id: uuid.UUID, owner: str, lease_s: int, progress: float | None = None,
              message: str | None = None) -> bool:
    """Extend the lease; returns True if cancellation has been requested."""
    vals: dict[str, Any] = {"lease_expires_at": now() + timedelta(seconds=lease_s), "updated_at": now()}
    if progress is not None:
        vals["progress"] = max(0.0, min(1.0, progress))
    if message is not None:
        vals["progress_message"] = message[:300]
    db.execute(update(Job).where(Job.id == job_id, Job.lease_owner == owner,
                                 Job.status.in_(("running", "cancellation_requested"))).values(**vals))
    db.commit()
    return bool(db.execute(select(Job.cancel_requested).where(Job.id == job_id)).scalar())


def succeed(db: Session, job: Job, result: dict[str, Any]) -> None:
    job.result = result
    job.progress = 1.0
    transition(db, job, "succeeded", "completed")


def fail(db: Session, job: Job, code: str, message: str, retryable: bool) -> str:
    job.error_code = code
    job.error_message = message[:4000]
    job.retryable = retryable
    if retryable and job.attempt < job.max_attempts and not job.cancel_requested:
        delay = BACKOFF_S[min(job.attempt - 1, len(BACKOFF_S) - 1)] * random.uniform(0.8, 1.2)
        job.next_retry_at = now() + timedelta(seconds=delay)
        job.lease_owner = None
        job.lease_expires_at = None
        transition(db, job, "retry_wait", f"{code}: retry in {delay:.1f}s")
        return "retry_wait"
    transition(db, job, "failed", f"{code}: {message[:300]}")
    return "failed"


def cancelled(db: Session, job: Job, reason: str = "cancelled cooperatively") -> None:
    transition(db, job, "cancelled", reason)


def request_cancel(db: Session, job: Job, reason: str) -> Job:
    """Idempotent: queued/retry_wait cancel immediately; running is flagged for cooperative stop."""
    if job.status in TERMINAL:
        return job
    job.cancel_requested = True
    if job.status in ("queued", "retry_wait"):
        transition(db, job, "cancelled", reason)
    elif job.status == "running":
        transition(db, job, "cancellation_requested", reason)
    return job


def requeue_due(db: Session, limit: int = 100) -> int:
    rows = db.execute(
        select(Job).where(Job.status == "retry_wait", Job.next_retry_at <= now()).limit(limit).with_for_update(
            skip_locked=True)
    ).scalars().all()
    for job in rows:
        transition(db, job, "queued", "retry due")
        enqueue_outbox(db, job)
    db.commit()
    return len(rows)


def recover_stale(db: Session, limit: int = 100) -> int:
    """Jobs whose worker died mid-run: lease expired -> retry (bounded) or fail."""
    rows = db.execute(
        select(Job).where(Job.status.in_(("running", "cancellation_requested")), Job.lease_expires_at < now())
        .limit(limit).with_for_update(skip_locked=True)
    ).scalars().all()
    for job in rows:
        if job.status == "cancellation_requested":
            transition(db, job, "cancelled", "lease expired after cancellation request")
        else:
            fail(db, job, "lease_expired", f"worker {job.lease_owner} stopped heartbeating", retryable=True)
    db.commit()
    return len(rows)


# ---- distributed leases ------------------------------------------------------------------------------------


def acquire_lease(db: Session, name: str, owner: str, ttl_s: int) -> bool:
    row = db.execute(
        text(
            "INSERT INTO leases(name, owner, acquired_at, heartbeat_at, expires_at) "
            "VALUES (:n, :o, now(), now(), now() + make_interval(secs => :t)) "
            "ON CONFLICT (name) DO UPDATE SET owner = EXCLUDED.owner, heartbeat_at = now(), "
            "expires_at = EXCLUDED.expires_at, "
            "acquired_at = CASE WHEN leases.owner = EXCLUDED.owner THEN leases.acquired_at ELSE now() END "
            "WHERE leases.expires_at < now() OR leases.owner = EXCLUDED.owner RETURNING owner"
        ),
        {"n": name, "o": owner, "t": ttl_s},
    ).first()
    db.commit()
    return row is not None and row[0] == owner


def release_lease(db: Session, name: str, owner: str) -> None:
    db.execute(text("DELETE FROM leases WHERE name = :n AND owner = :o"), {"n": name, "o": owner})
    db.commit()


def lease_holder(db: Session, name: str) -> dict[str, Any] | None:
    row = db.execute(text("SELECT owner, acquired_at, expires_at FROM leases WHERE name = :n AND expires_at > now()"),
                     {"n": name}).first()
    return dict(row._mapping) if row else None
