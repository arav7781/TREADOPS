"""One Celery task, `treadops.run_job`, dispatches by job kind. Status lives in SQL, not in Celery."""

from __future__ import annotations

import logging
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from celery.signals import worker_ready
from sqlalchemy.exc import OperationalError

from treadops_api import jobs
from treadops_api.db import new_session
from treadops_api.models import Job
from treadops_api.storage import StorageUnavailable
from treadops_ml.bundle import BundleInvalid
from treadops_ml.preprocess import ImageRejected

from .celery_app import app
from .runtime import WORKER_ID, start_heartbeat

log = logging.getLogger("treadops.worker")

LEASE_S = {"training": 900, "replay": 300, "shadow": 600, "snapshot": 600, "promotion": 300, "rollback": 300}
DEFAULT_LEASE_S = 120


class Permanent(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Transient(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class JobCancelled(Exception):
    pass


@dataclass
class JobContext:
    job_id: uuid.UUID
    kind: str
    payload: dict[str, Any]
    attempt: int
    correlation_id: str | None
    created_by: uuid.UUID | None
    owner: str = WORKER_ID
    log_lines: list[str] = field(default_factory=list)

    def heartbeat(self, progress: float | None = None, message: str | None = None) -> None:
        db = new_session()
        try:
            if jobs.heartbeat(db, self.job_id, self.owner, LEASE_S.get(self.kind, DEFAULT_LEASE_S), progress, message):
                raise JobCancelled()
        finally:
            db.close()

    def job(self, db) -> Job:
        return db.get(Job, self.job_id, populate_existing=True)


Handler = Callable[[JobContext], dict[str, Any]]
HANDLERS: dict[str, Handler] = {}
ON_FAIL: dict[str, Callable[[Any, Job, str], None]] = {}


def handler(kind: str):
    def deco(fn: Handler) -> Handler:
        HANDLERS[kind] = fn
        return fn

    return deco


def on_fail(kind: str):
    def deco(fn):
        ON_FAIL[kind] = fn
        return fn

    return deco


def _classify(exc: Exception) -> tuple[str, str, bool]:
    if isinstance(exc, Permanent):
        return exc.code, str(exc), False
    if isinstance(exc, Transient):
        return exc.code, str(exc), True
    if isinstance(exc, BundleInvalid):
        return f"bundle_{exc.code}", str(exc), False
    if isinstance(exc, ImageRejected):
        return exc.code, exc.message, False
    if isinstance(exc, StorageUnavailable):
        return "storage_unavailable", str(exc), True
    if isinstance(exc, OperationalError):
        return "database_unavailable", str(exc).splitlines()[0], True
    if isinstance(exc, ConnectionError | TimeoutError):
        return "dependency_unavailable", str(exc), True
    return "unexpected_error", f"{exc.__class__.__name__}: {exc}", True


@app.task(name="treadops.run_job")
def run_job(job_id: str) -> None:
    from . import handlers  # noqa: F401  (registers HANDLERS)

    jid = uuid.UUID(job_id)
    db = new_session()
    try:
        peek = db.get(Job, jid)
        if peek is None:
            log.warning("job %s not found", job_id)
            return
        job = jobs.claim(db, jid, WORKER_ID, LEASE_S.get(peek.kind, DEFAULT_LEASE_S))
        if job is None:
            log.info("job %s not claimable (status=%s); duplicate delivery ignored", job_id, peek.status)
            return
        ctx = JobContext(job.id, job.kind, dict(job.payload), job.attempt, job.correlation_id, job.created_by)
    finally:
        db.close()

    try:
        HANDLERS[ctx.kind](ctx)
        log.info("job %s (%s) finished", job_id, ctx.kind)
    except JobCancelled:
        db = new_session()
        try:
            job = ctx.job(db)
            if job.status not in jobs.TERMINAL:
                if job.status == "running":
                    jobs.transition(db, job, "cancellation_requested", "cancel observed")
                jobs.cancelled(db, job)
                if ctx.kind in ON_FAIL:
                    ON_FAIL[ctx.kind](db, job, "cancelled")
                db.commit()
        finally:
            db.close()
    except Exception as exc:
        code, message, retryable = _classify(exc)
        log.warning("job %s (%s) error %s: %s\n%s", job_id, ctx.kind, code, message, traceback.format_exc(limit=6))
        db = new_session()
        try:
            job = ctx.job(db)
            if job.status in jobs.TERMINAL:
                return
            outcome = jobs.fail(db, job, code, message, retryable)
            if outcome == "failed" and ctx.kind in ON_FAIL:
                ON_FAIL[ctx.kind](db, job, "failed")
            db.commit()
        finally:
            db.close()


@worker_ready.connect
def _on_ready(sender=None, **_):
    queues = ",".join(q.name for q in sender.app.amqp.queues.consume_from.values()) if sender else "unknown"
    start_heartbeat(queues)
