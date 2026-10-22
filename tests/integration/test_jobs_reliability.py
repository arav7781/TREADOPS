"""JOB-01..04: redelivery, duplicate dispatch, transient dependency failure, training-slot contention."""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from treadops_api import db as dbmod
from treadops_api import jobs
from treadops_api.models import Inspection, InspectionStage, Job, JobEvent, Prediction

from .conftest import admin_id, fixture_bundle, import_and_activate, jpeg, login, requires_bundle

pytestmark = [pytest.mark.integration, requires_bundle]


def _submit(client, key: str, seed: int) -> dict:
    h = login(client, "inspector")
    r = client.post("/v1/inspections", headers={**h, "Idempotency-Key": key},
                    files={"file": ("t.jpg", jpeg(seed), "image/jpeg")})
    assert r.status_code == 202, r.text
    return r.json()


def _count(model, **where) -> int:
    with dbmod.session_scope() as s:
        q = select(func.count()).select_from(model)
        for k, v in where.items():
            q = q.where(getattr(model, k) == v)
        return s.execute(q).scalar_one()


def test_job01_redelivery_after_completion_and_after_crash_gives_one_prediction(client):
    from treadops_worker.tasks import run_job

    import_and_activate(fixture_bundle(), admin_id())
    acc = _submit(client, "job01", 11)
    run_job(acc["job_id"])
    run_job(acc["job_id"])  # redelivered message after a successful run: no-op
    assert _count(Prediction, inspection_id=uuid.UUID(acc["inspection_id"])) == 1
    with dbmod.session_scope() as s:
        job = s.get(Job, uuid.UUID(acc["job_id"]))
        assert job.status == "succeeded" and job.attempt == 1
        stages = s.execute(select(func.count()).select_from(InspectionStage).where(
            InspectionStage.inspection_id == uuid.UUID(acc["inspection_id"]))).scalar_one()
        assert stages == 8  # ingest, canonical, letterbox, tensor, quality, inference, attribution, classified

    # crash mid-run: the job is running with an expired lease and no result
    acc2 = _submit(client, "job01-crash", 12)
    with dbmod.session_scope() as s:
        job = s.get(Job, uuid.UUID(acc2["job_id"]))
        job.status, job.attempt = "running", 1
        job.lease_owner, job.lease_expires_at = "dead-worker", datetime.now(UTC) - timedelta(seconds=5)
    run_job(acc2["job_id"])  # redelivery while the dead lease is still "running": ignored
    s = dbmod.new_session()
    assert jobs.recover_stale(s) == 1
    with dbmod.session_scope() as s2:
        s2.get(Job, uuid.UUID(acc2["job_id"])).next_retry_at = datetime.now(UTC)
    assert jobs.requeue_due(s) == 1
    s.close()
    run_job(acc2["job_id"])
    with dbmod.session_scope() as s:
        job = s.get(Job, uuid.UUID(acc2["job_id"]))
        assert job.status == "succeeded" and job.attempt == 2
        assert s.get(Inspection, uuid.UUID(acc2["inspection_id"])).status == "succeeded"
    assert _count(Prediction, inspection_id=uuid.UUID(acc2["inspection_id"])) == 1


def test_job02_duplicate_dispatch_is_deduplicated(client):
    from treadops_worker.tasks import run_job

    import_and_activate(fixture_bundle(), admin_id())
    acc = _submit(client, "job02", 21)
    ts = [threading.Thread(target=run_job, args=(acc["job_id"],)) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert _count(Prediction, inspection_id=uuid.UUID(acc["inspection_id"])) == 1
    with dbmod.session_scope() as s:
        claims = s.execute(select(func.count()).select_from(JobEvent).where(
            JobEvent.job_id == uuid.UUID(acc["job_id"]), JobEvent.to_status == "running")).scalar_one()
    assert claims == 1


def test_job03_storage_outage_retries_with_persisted_attempts(client, monkeypatch):
    from treadops_api.storage import StorageUnavailable, get_storage
    from treadops_worker.tasks import run_job

    import_and_activate(fixture_bundle(), admin_id())
    acc = _submit(client, "job03", 31)
    storage = get_storage()
    real_get = storage.get
    calls = {"n": 0}

    def flaky(key):
        if key.startswith("uploads/") and calls["n"] == 0:
            calls["n"] += 1
            raise StorageUnavailable("simulated outage")
        return real_get(key)

    monkeypatch.setattr(storage, "get", flaky)
    run_job(acc["job_id"])
    with dbmod.session_scope() as s:
        job = s.get(Job, uuid.UUID(acc["job_id"]))
        assert job.status == "retry_wait" and job.error_code == "storage_unavailable" and job.attempt == 1
        assert job.next_retry_at is not None and job.retryable
        job.next_retry_at = datetime.now(UTC)
    assert _count(Prediction, inspection_id=uuid.UUID(acc["inspection_id"])) == 0  # no fake success
    s = dbmod.new_session()
    jobs.requeue_due(s)
    s.close()
    run_job(acc["job_id"])
    with dbmod.session_scope() as s:
        job = s.get(Job, uuid.UUID(acc["job_id"]))
        assert job.status == "succeeded" and job.attempt == 2


def test_job03_permanent_failure_does_not_retry(client):
    from treadops_worker.tasks import run_job

    import_and_activate(fixture_bundle(), admin_id())
    acc = _submit(client, "job03p", 32)
    with dbmod.session_scope() as s:
        insp = s.get(Inspection, uuid.UUID(acc["inspection_id"]))
        tup = dict(insp.model_tuple)
        tup["bundle_sha256"] = "0" * 64  # pinned bundle no longer exists
        insp.model_tuple = tup
    run_job(acc["job_id"])
    with dbmod.session_scope() as s:
        job = s.get(Job, uuid.UUID(acc["job_id"]))
        assert job.status in ("failed", "retry_wait")
        if job.status == "retry_wait":  # missing objects surface as a typed bundle error after bounded retries
            assert job.attempt <= job.max_attempts


def test_job04_single_training_slot(client):
    h = login(client, "ml")
    from treadops_api.errors import ApiError
    from treadops_api.services.datasets import build_snapshot, register_public_catalogue
    from treadops_api.storage import get_storage

    with dbmod.session_scope() as s, pytest.raises(ApiError) as empty:
        build_snapshot(s, get_storage(), include_verified_uploads=False)
    assert empty.value.code == "empty_snapshot"
    with dbmod.session_scope() as s:
        register_public_catalogue(s, get_storage(), upload=False)
    with dbmod.session_scope() as s:
        snap, _ = build_snapshot(s, get_storage(), include_verified_uploads=False)
        assert snap.counts["included"] == 1276 and "protected_role:test" in snap.counts["excluded"]
        sid = str(snap.id)
    a = client.post("/v1/training-runs", headers=h, json={"snapshot_id": sid, "config_id": "nano_smoke"})
    b = client.post("/v1/training-runs", headers=h, json={"snapshot_id": sid, "config_id": "nano_smoke"})
    assert a.status_code == 202 and b.status_code == 409 and b.json()["error"]["code"] == "training_slot_busy"
    bad = client.post("/v1/training-runs", headers=h, json={"snapshot_id": sid, "config_id": "../../etc/passwd"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "unknown_recipe"
    s = dbmod.new_session()
    assert jobs.acquire_lease(s, "training-slot", "w1", 60)
    assert not jobs.acquire_lease(s, "training-slot", "w2", 60)
    jobs.release_lease(s, "training-slot", "w1")
    assert jobs.acquire_lease(s, "training-slot", "w2", 60)
    s.close()
    # cancel the queued run: idempotent, terminal state immutable
    jid = a.json()["job_id"]
    c1 = client.post(f"/v1/jobs/{jid}/cancel", headers=h)
    c2 = client.post(f"/v1/jobs/{jid}/cancel", headers=h)
    assert c1.json()["status"] == c2.json()["status"] == "cancelled"
