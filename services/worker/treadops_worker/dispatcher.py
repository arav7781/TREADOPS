"""Outbox dispatcher + recovery sweeper + scheduler (one active instance via the `scheduler` lease).

A crash between publish and marking `dispatched` re-publishes the event; the SQL claim in `run_job`
deduplicates it. Publishing never happens inside the request transaction.
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, text

from treadops_api import jobs
from treadops_api.db import new_session
from treadops_api.models import Inspection, MonitoringWindow, OutboxEvent

from .celery_app import app
from .runtime import WORKER_ID

log = logging.getLogger("treadops.dispatcher")
_running = True


def publish_pending(batch: int = 50) -> int:
    db = new_session()
    try:
        rows = db.execute(
            select(OutboxEvent).where(OutboxEvent.dispatch_state == "pending",
                                      OutboxEvent.available_at <= func.now())
            .order_by(OutboxEvent.created_at).limit(batch).with_for_update(skip_locked=True)
        ).scalars().all()
        for ev in rows:
            try:
                if ev.topic == "job.dispatch":
                    app.send_task("treadops.run_job", args=[ev.payload["job_id"]], queue=ev.payload["queue"],
                                  task_id=f"{ev.payload['job_id']}:{ev.id}")
                ev.dispatch_state = "dispatched"
                ev.dispatched_at = datetime.now(UTC)
            except Exception as exc:  # broker down: back off this event, keep the rest of the batch
                ev.attempts += 1
                ev.last_error = f"{exc.__class__.__name__}: {exc}"[:2000]
                ev.available_at = datetime.now(UTC) + timedelta(seconds=min(60, 2 ** ev.attempts))
        db.commit()
        return len(rows)
    finally:
        db.close()


def sweep() -> dict[str, int]:
    db = new_session()
    try:
        return {"requeued": jobs.requeue_due(db), "recovered": jobs.recover_stale(db)}
    finally:
        db.close()


def maybe_schedule_monitoring(every: int = 200) -> str | None:
    """Enqueue a monitoring window job every `every` completed inspections (durable dedupe key)."""
    db = new_session()
    try:
        done = db.execute(select(func.count()).select_from(Inspection)
                          .where(Inspection.status == "succeeded")).scalar_one()
        bucket = done // every
        if bucket == 0:
            return None
        key = f"monitoring:completions:{bucket * every}"
        if db.execute(text("SELECT 1 FROM jobs WHERE dedupe_key = :k"), {"k": key}).first():
            return None
        last = db.execute(select(func.max(MonitoringWindow.created_at))).scalar_one_or_none()
        jobs.create_job(db, kind="monitoring", dedupe_key=key, payload={"trigger": "completions", "completed": done,
                                                                       "previous_window_at": str(last)})
        db.commit()
        return key
    finally:
        db.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    def stop(*_):
        global _running
        _running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    owner = f"dispatcher:{WORKER_ID}"
    last_sweep = last_sched = 0.0
    log.info("dispatcher %s started", owner)
    while _running:
        try:
            n = publish_pending()
            now = time.monotonic()
            if now - last_sweep > 5:
                db = new_session()
                leader = jobs.acquire_lease(db, "scheduler", owner, ttl_s=30)
                db.close()
                if leader:
                    s = sweep()
                    if any(s.values()):
                        log.info("sweep %s", s)
                    if now - last_sched > 30:
                        key = maybe_schedule_monitoring()
                        if key:
                            log.info("scheduled %s", key)
                        last_sched = now
                last_sweep = now
            time.sleep(0.2 if n else 0.5)
        except Exception:
            log.exception("dispatcher loop error; backing off")
            time.sleep(3)
    log.info("dispatcher stopped")


if __name__ == "__main__":
    main()
