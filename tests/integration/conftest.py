"""Integration profile: real PostgreSQL (treadops_test), Redis and object storage (bucket treadops-test).

Settings are cached process-wide, so the environment is switched before any application module is used and
every cache is cleared. Tests call worker handlers in-process to make redelivery/crash scenarios deterministic.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import uuid
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    test_url = os.environ.get("TEST_DATABASE_URL")
    if not test_url:
        pytest.skip("TEST_DATABASE_URL not set (run `make env` and `make infra`)", allow_module_level=True)
    os.environ["DATABASE_URL"] = test_url
    os.environ["S3_BUCKET"] = "treadops-test"
    os.environ["INFERENCE_QUEUE_LIMIT"] = "1000"
    os.environ["BUNDLE_CACHE_DIR"] = "models/cache"


_load_env()

from treadops_api import db as dbmod  # noqa: E402
from treadops_api import settings as setmod  # noqa: E402
from treadops_api import storage as stmod  # noqa: E402

setmod.get_settings.cache_clear()
dbmod.get_engine.cache_clear()
dbmod._factory.cache_clear()
stmod.get_storage.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from treadops_api.models import User  # noqa: E402
from treadops_api.security import hash_password  # noqa: E402

PASSWORD = "integration-" + uuid.uuid4().hex[:8]
BUNDLE_DIR = ROOT / "artifacts/bundle"


@pytest.fixture(scope="session")
def migrated():
    """DB-01: migrate an empty database to head."""
    from alembic import command
    from alembic.config import Config

    eng = dbmod.get_engine()
    with eng.begin() as c:
        c.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    command.upgrade(cfg, "head")
    stmod.get_storage().ensure_bucket()
    return cfg


@pytest.fixture()
def clean(migrated):
    eng = dbmod.get_engine()
    with eng.begin() as c:
        tables = [r[0] for r in c.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename <> 'alembic_version'"))]
        c.execute(text("TRUNCATE " + ", ".join(f'"{t}"' for t in tables) + " RESTART IDENTITY CASCADE"))
    with dbmod.session_scope() as s:
        for email, roles in (("viewer@t.local", ["viewer"]), ("inspector@t.local", ["viewer", "inspector"]),
                             ("ml@t.local", ["viewer", "ml_engineer"]),
                             ("admin@t.local", ["viewer", "inspector", "ml_engineer", "admin"])):
            s.add(User(email=email, display_name=email.split("@")[0], password_hash=hash_password(PASSWORD),
                       roles=roles, is_fixture=True))
    yield


@pytest.fixture()
def client(clean):
    from treadops_api.main import create_app

    return TestClient(create_app())


def login(client, who: str) -> dict:
    r = client.post("/v1/auth/login", json={"email": f"{who}@t.local", "password": PASSWORD})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def jpeg(seed: int = 0, size=(640, 480)) -> bytes:
    import numpy as np

    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


requires_bundle = pytest.mark.skipif(not (BUNDLE_DIR / "manifest.json").exists(),
                                     reason="no trusted bundle: run the DVC pipeline (make train evaluate ...)")


def variant_bundle(tmp: Path, *, low: float, high: float, version: str) -> Path:
    """A second, distinct trusted bundle: same weights, different versioned triage policy."""
    from treadops_ml.bundle import bundle_sha_from_files
    from treadops_ml.common import sha256_file

    dst = tmp / f"bundle-{version}"
    shutil.copytree(BUNDLE_DIR, dst)
    pol = json.loads((dst / "policy.json").read_text())
    pol.update(triage_low=low, triage_high=high, version=version)
    (dst / "policy.json").write_text(json.dumps(pol, indent=2, sort_keys=True))
    golden = json.loads((dst / "golden.json").read_text())
    from treadops_ml.policy import Policy

    p = Policy.from_dict(pol)
    for item in golden["items"]:
        item["triage"] = p.decide(item["p_defective"])[0]
    (dst / "golden.json").write_text(json.dumps(golden, indent=2, sort_keys=True))
    m = json.loads((dst / "manifest.json").read_text())
    m["files"] = {k: sha256_file(dst / k) for k in m["files"]}
    m["bundle_sha256"] = bundle_sha_from_files(m["files"])
    m["serving"]["policy_version"] = version
    (dst / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True))
    return dst


def import_and_activate(bundle_dir: Path, admin_id) -> tuple:
    """Import a bundle and switch the pointer through the real release handler (stage, warm, CAS)."""
    from treadops_api import jobs
    from treadops_api.services.inspections import active_deployment
    from treadops_api.services.releases import import_bundle
    from treadops_worker.tasks import run_job

    with dbmod.session_scope() as s:
        mv = import_bundle(s, stmod.get_storage(), bundle_dir)
        ptr, _ = active_deployment(s)
        gen = ptr.generation
        job = jobs.create_job(s, kind="promotion", dedupe_key=f"promote:{uuid.uuid4()}", max_attempts=1,
                              payload={"model_version_id": str(mv.id), "expected_generation": gen,
                                       "reason": "integration test activation", "approved_by": str(admin_id)})
        mv_id, job_id = mv.id, job.id
    run_job(str(job_id))
    return mv_id, job_id


def admin_id():
    from sqlalchemy import select

    with dbmod.session_scope() as s:
        return s.execute(select(User.id).where(User.email == "admin@t.local")).scalar_one()
