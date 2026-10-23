"""REL-01..04: version pinning across promotion, failed warm-up, concurrent promotion, rollback + registry outage."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from treadops_api import db as dbmod
from treadops_api import jobs
from treadops_api.models import AuditEvent, Deployment, EnvironmentPointer, Job, ModelVersion, Prediction
from treadops_api.services.inspections import active_deployment

from .conftest import (
    admin_id,
    fixture_bundle,
    import_and_activate,
    jpeg,
    login,
    requires_bundle,
    variant_bundle,
)

pytestmark = [pytest.mark.integration, requires_bundle]


def _pointer():
    with dbmod.session_scope() as s:
        ptr, dep = active_deployment(s)
        return ptr.generation, dep.bundle_sha256 if dep else None


def _promote_job(mv_id, gen) -> str:
    with dbmod.session_scope() as s:
        j = jobs.create_job(s, kind="promotion", dedupe_key=f"p:{uuid.uuid4()}", max_attempts=1,
                            payload={"model_version_id": str(mv_id), "expected_generation": gen,
                                     "reason": "integration promotion", "approved_by": str(admin_id())})
        return str(j.id)


def test_rel01_queued_job_keeps_its_pinned_bundle_across_promotion(client, tmp_path):
    from treadops_worker.tasks import run_job

    import_and_activate(fixture_bundle(), admin_id())
    gen1, sha1 = _pointer()
    h = login(client, "inspector")
    old = client.post("/v1/inspections", headers={**h, "Idempotency-Key": "rel01-old"},
                      files={"file": ("a.jpg", jpeg(41), "image/jpeg")}).json()
    v2 = variant_bundle(tmp_path, low=0.1, high=0.9, version="v2-test")
    import_and_activate(v2, admin_id())
    gen2, sha2 = _pointer()
    assert gen2 == gen1 + 1 and sha2 != sha1
    new = client.post("/v1/inspections", headers={**h, "Idempotency-Key": "rel01-new"},
                      files={"file": ("b.jpg", jpeg(42), "image/jpeg")}).json()
    assert old["deployment_generation"] == gen1 and new["deployment_generation"] == gen2
    run_job(old["job_id"])
    run_job(new["job_id"])
    with dbmod.session_scope() as s:
        p_old = s.execute(select(Prediction).where(Prediction.inspection_id == uuid.UUID(old["inspection_id"]))).scalar_one()
        p_new = s.execute(select(Prediction).where(Prediction.inspection_id == uuid.UUID(new["inspection_id"]))).scalar_one()
    assert p_old.lineage["bundle_sha256"] == sha1 and p_old.lineage["policy_version"] != "v2-test"
    assert p_new.lineage["bundle_sha256"] == sha2 and p_new.lineage["policy_version"] == "v2-test"


def test_rel02_tampered_bundle_fails_warmup_and_leaves_pointer(client, tmp_path):
    from treadops_api.services.releases import import_bundle
    from treadops_api.storage import get_storage
    from treadops_worker.tasks import run_job

    import_and_activate(fixture_bundle(), admin_id())
    gen, sha = _pointer()
    v3 = variant_bundle(tmp_path, low=0.15, high=0.85, version="v3-test")
    with dbmod.session_scope() as s:
        mv = import_bundle(s, get_storage(), v3)
        mv_id, bsha = mv.id, mv.bundle_sha256
    get_storage().put(f"bundles/{bsha}/calibration.json", b'{"temperature": 99}')  # corrupt after import
    import shutil

    from treadops_ml.common import repo_path
    shutil.rmtree(repo_path("models/cache") / bsha, ignore_errors=True)
    jid = _promote_job(mv_id, gen)
    run_job(jid)
    with dbmod.session_scope() as s:
        job = s.get(Job, uuid.UUID(jid))
        assert job.status == "failed" and "hash_mismatch" in (job.error_code or "") + (job.error_message or "")
        failed = s.execute(select(AuditEvent).where(AuditEvent.action == "release.promotion.failed")).scalars().all()
        assert failed and failed[-1].payload["pointer"] == "unchanged"
    assert _pointer() == (gen, sha)


def test_rel03_concurrent_promotions_one_winner(client, tmp_path):
    from treadops_api.services.releases import import_bundle
    from treadops_api.storage import get_storage
    from treadops_worker.tasks import run_job

    import_and_activate(fixture_bundle(), admin_id())
    gen, _ = _pointer()
    ids = []
    for v in ("v4a-test", "v4b-test"):
        with dbmod.session_scope() as s:
            ids.append(import_bundle(s, get_storage(), variant_bundle(tmp_path, low=0.18, high=0.82 if v.endswith("a-test")
                                                                       else 0.83, version=v)).id)
    jobs_ = [_promote_job(i, gen) for i in ids]
    for j in jobs_:
        run_job(j)
    with dbmod.session_scope() as s:
        statuses = sorted(s.get(Job, uuid.UUID(j)).status for j in jobs_)
        loser = next(s.get(Job, uuid.UUID(j)) for j in jobs_ if s.get(Job, uuid.UUID(j)).status == "failed")
        assert statuses == ["failed", "succeeded"] and loser.error_code == "generation_conflict"
        assert s.execute(select(EnvironmentPointer)).scalar_one().generation == gen + 1
    h = login(client, "admin")
    stale = client.post("/v1/deployments/promote", headers=h,
                        json={"model_version_id": str(ids[0]), "expected_generation": gen, "reason": "stale request"})
    assert stale.status_code == 409 and stale.json()["error"]["code"] in ("generation_conflict", "already_active")


def test_rel04_rollback_restores_previous_bundle_and_registry_reconciles(client, tmp_path, monkeypatch):
    from treadops_worker.tasks import run_job

    first_mv, _ = import_and_activate(fixture_bundle(), admin_id())
    gen1, sha1 = _pointer()
    import_and_activate(variant_bundle(tmp_path, low=0.12, high=0.88, version="v5-test"), admin_id())
    gen2, sha2 = _pointer()
    h = login(client, "admin")
    blocked = client.post("/v1/deployments/promote", headers=h,
                          json={"model_version_id": str(uuid.uuid4()), "expected_generation": gen2, "reason": "x" * 10})
    assert blocked.status_code == 404
    r = client.post("/v1/deployments/rollback", headers=h,
                    json={"model_version_id": str(first_mv), "expected_generation": gen2,
                          "reason": "post-release regression drill"})
    assert r.status_code == 202, r.text
    run_job(r.json()["job_id"])
    gen3, sha3 = _pointer()
    assert gen3 == gen2 + 1 and sha3 == sha1
    with dbmod.session_scope() as s:
        dep = s.execute(select(Deployment).where(Deployment.generation == gen3)).scalar_one()
        assert dep.action == "rollback" and dep.previous_deployment_id is not None
        assert s.get(ModelVersion, first_mv).stage == "active"
        recon = s.execute(select(Job).where(Job.kind == "registry_reconcile", Job.dedupe_key ==
                                            f"registry:champion:gen{gen3}")).scalar_one()
        recon_id = str(recon.id)
        s.get(ModelVersion, first_mv).registry_version = s.get(ModelVersion, first_mv).registry_version or "1"
        recon.payload = {**recon.payload, "registry_version": "1"}

    import mlflow

    class Down:
        def __init__(self, *a, **k):
            raise ConnectionError("mlflow unreachable")

    monkeypatch.setattr(mlflow, "MlflowClient", Down)
    run_job(recon_id)
    with dbmod.session_scope() as s:
        j = s.get(Job, uuid.UUID(recon_id))
        assert j.status == "retry_wait" and j.error_code == "registry_unavailable"
    assert _pointer() == (gen3, sha1)  # serving pointer is authoritative even while the registry is down

    calls = []

    class Up:
        def __init__(self, *a, **k):
            pass

        def set_registered_model_alias(self, *a):
            calls.append(("alias", a))

        def set_model_version_tag(self, *a):
            calls.append(("tag", a))

    monkeypatch.setattr(mlflow, "MlflowClient", Up)
    from datetime import UTC, datetime

    with dbmod.session_scope() as s:
        s.get(Job, uuid.UUID(recon_id)).next_retry_at = datetime.now(UTC)
    s = dbmod.new_session()
    jobs.requeue_due(s)
    s.close()
    run_job(recon_id)
    with dbmod.session_scope() as s:
        assert s.get(Job, uuid.UUID(recon_id)).status == "succeeded"
    assert any(c[0] == "alias" and c[1][1] == "champion" for c in calls)


def test_rel_api_promotion_happy_path_and_failed_gate_block(client):
    """The admin endpoint creates a promotion job the worker can run; failing gates are refused up front."""
    import json
    import shutil

    from treadops_api.services.releases import import_bundle
    from treadops_api.storage import get_storage
    from treadops_worker.tasks import run_job

    from .conftest import fixture_bundle

    with dbmod.session_scope() as s:
        mv_id = import_bundle(s, get_storage(), fixture_bundle()).id
    h = login(client, "admin")
    r = client.post("/v1/deployments/promote", headers=h,
                    json={"model_version_id": str(mv_id), "expected_generation": 0,
                          "reason": "fixture passes its gates in the test database"})
    assert r.status_code == 202, r.text
    run_job(r.json()["job_id"])
    gen, sha = _pointer()
    assert gen == 1 and sha
    detail = client.get(f"/v1/jobs/{r.json()['job_id']}", headers=h).json()
    assert detail["kind"] == "promotion" and detail["status"] == "succeeded"
    assert detail["result"]["readiness"]["status"] == "ready"

    failing = fixture_bundle().parent / "bundle-failing"
    if failing.exists():
        shutil.rmtree(failing)
    shutil.copytree(fixture_bundle(), failing)
    gate = json.loads((failing / "gate_report.json").read_text())
    gate["status"] = "failed"
    (failing / "gate_report.json").write_text(json.dumps(gate))
    from .conftest import _rehash

    _rehash(failing)
    with dbmod.session_scope() as s:
        bad = import_bundle(s, get_storage(), failing)
        bad_id, stage = bad.id, bad.stage
    assert stage == "rejected"
    blocked = client.post("/v1/deployments/promote", headers=h,
                          json={"model_version_id": str(bad_id), "expected_generation": 1,
                                "reason": "should be refused by the gate"})
    assert blocked.status_code == 409 and blocked.json()["error"]["code"] == "gate_not_passed"
    with dbmod.session_scope() as s:
        assert s.execute(select(AuditEvent).where(AuditEvent.action == "release.blocked")).scalars().first()
    assert _pointer() == (gen, sha)
