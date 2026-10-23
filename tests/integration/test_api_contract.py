"""API-01..04 and DB-01 against real PostgreSQL / object storage."""

from __future__ import annotations

import threading
import uuid

import pytest
from sqlalchemy import func, select

from treadops_api import db as dbmod
from treadops_api.models import Inspection, Job, ReviewLabel

from .conftest import admin_id, fixture_bundle, import_and_activate, jpeg, login, requires_bundle

pytestmark = pytest.mark.integration


def _jobs() -> int:
    with dbmod.session_scope() as s:
        return s.execute(select(func.count()).select_from(Job)).scalar_one()


def test_db01_migration_is_current_and_round_trips(migrated):
    from alembic import command
    from alembic.script import ScriptDirectory
    from sqlalchemy import text

    with dbmod.get_engine().connect() as c:
        current = c.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert current == ScriptDirectory.from_config(migrated).get_current_head()
    command.downgrade(migrated, "base")
    command.upgrade(migrated, "head")
    command.check(migrated)  # raises if models and migrations drift


def test_api02_missing_token_and_wrong_role(client):
    assert client.get("/v1/me").status_code == 401
    assert client.get("/v1/me").json()["error"]["code"] == "missing_token"
    viewer = login(client, "viewer")
    r = client.post("/v1/inspections", headers={**viewer, "Idempotency-Key": "k1"},
                    files={"file": ("t.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
    r = client.post("/v1/deployments/promote", headers=viewer,
                    json={"model_version_id": str(uuid.uuid4()), "expected_generation": 0, "reason": "not allowed here"})
    assert r.status_code == 403
    assert client.get("/v1/audit-events", headers=viewer).status_code == 403
    assert _jobs() == 0  # no mutation happened
    bad = client.get("/v1/me", headers={"Authorization": "Bearer not-a-jwt"})
    assert bad.status_code == 401 and "Traceback" not in bad.text


def test_logout_revokes_the_session_immediately(client):
    h = login(client, "inspector")
    assert client.get("/v1/me", headers=h).status_code == 200
    assert client.post("/v1/auth/logout", headers=h).status_code == 200
    r = client.get("/v1/me", headers=h)
    assert r.status_code == 401 and r.json()["error"]["code"] == "session_revoked"


def test_upload_without_a_model_is_explicitly_unavailable(client):
    h = login(client, "inspector")
    r = client.post("/v1/inspections", headers={**h, "Idempotency-Key": "k-nomodel"},
                    files={"file": ("t.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"
    assert r.headers.get("retry-after") and _jobs() == 0


@requires_bundle
@pytest.mark.parametrize("payload,ctype,status,code", [
    (b"", "image/jpeg", 400, "empty_upload"),
    (b"definitely not an image", "image/jpeg", 415, "unsupported_media"),
    (b"\xff\xd8\xff\xe0" + b"\x00" * 64, "image/jpeg", 415, "unsupported_media"),
])
def test_api01_bad_uploads_get_typed_4xx_and_no_job(client, payload, ctype, status, code):
    import_and_activate(fixture_bundle(), admin_id())
    h = login(client, "inspector")
    before = _jobs()
    r = client.post("/v1/inspections", headers={**h, "Idempotency-Key": f"bad-{code}-{len(payload)}"},
                    files={"file": ("x.jpg", payload, ctype)})
    assert r.status_code == status and r.json()["error"]["code"] == code
    assert r.json()["error"]["retryable"] is False and r.json()["error"]["request_id"]
    assert _jobs() == before


@requires_bundle
def test_api01_oversized_upload_is_413(client, monkeypatch):
    import_and_activate(fixture_bundle(), admin_id())
    from treadops_api.settings import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_bytes", 1000)
    h = login(client, "inspector")
    r = client.post("/v1/inspections", headers={**h, "Idempotency-Key": "big"},
                    files={"file": ("x.jpg", jpeg(size=(800, 600)), "image/jpeg")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"


@requires_bundle
def test_api03_idempotency_key_replays_or_conflicts(client):
    import_and_activate(fixture_bundle(), admin_id())
    h = {**login(client, "inspector"), "Idempotency-Key": "same-key"}
    a = client.post("/v1/inspections", headers=h, files={"file": ("a.jpg", jpeg(1), "image/jpeg")})
    b = client.post("/v1/inspections", headers=h, files={"file": ("a.jpg", jpeg(1), "image/jpeg")})
    assert a.status_code == b.status_code == 202
    assert a.json()["inspection_id"] == b.json()["inspection_id"]
    assert b.headers["idempotent-replay"] == "true" and a.headers["location"].endswith(a.json()["inspection_id"])
    c = client.post("/v1/inspections", headers=h, files={"file": ("c.jpg", jpeg(2), "image/jpeg")})
    assert c.status_code == 409 and c.json()["error"]["code"] == "idempotency_conflict"
    with dbmod.session_scope() as s:
        assert s.execute(select(func.count()).select_from(Inspection)).scalar_one() == 1


@requires_bundle
def test_api04_concurrent_reviews_one_winner(client):
    import_and_activate(fixture_bundle(), admin_id())
    h = login(client, "inspector")
    iid = client.post("/v1/inspections", headers={**h, "Idempotency-Key": "rev"},
                      files={"file": ("r.jpg", jpeg(3), "image/jpeg")}).json()["inspection_id"]
    results: list[int] = []

    def review(label):
        r = client.post(f"/v1/inspections/{iid}/reviews", headers=h,
                        json={"expected_revision": 0, "label": label, "reason": f"looks {label}"})
        results.append(r.status_code)

    ts = [threading.Thread(target=review, args=(lab,)) for lab in ("good", "defective")]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(results) == [201, 409]
    with dbmod.session_scope() as s:
        assert s.execute(select(func.count()).select_from(ReviewLabel)).scalar_one() == 1
    stale = client.post(f"/v1/inspections/{iid}/reviews", headers=h,
                        json={"expected_revision": 0, "label": "good", "reason": "late"})
    assert stale.status_code == 409 and stale.json()["error"]["details"]["current_revision"] == 1
    ok = client.post(f"/v1/inspections/{iid}/reviews", headers=h,
                     json={"expected_revision": 1, "label": "unjudgeable", "reason": "reconciled after reload",
                           "disposition": "reinspect"})
    assert ok.status_code == 201 and ok.json()["revision"] == 2


def test_list_endpoints_enforce_page_size(client):
    h = login(client, "viewer")
    assert client.get("/v1/inspections?limit=101", headers=h).status_code == 422
    assert client.get("/v1/inspections?limit=100", headers=h).status_code == 200


def test_aggregate_endpoints_return_200_with_data(client):
    """Regression: the ledger GROUP BY on a JSON key and the LLMOps boolean cast both returned 500."""
    h = login(client, "admin")
    for path in ("/v1/circularity/ledger", "/v1/llmops/overview", "/v1/metrics/overview", "/v1/monitoring/windows"):
        r = client.get(path, headers=h)
        assert r.status_code == 200, (path, r.text[:300])
    assert "psi_alert_candidate" in client.get("/v1/monitoring/windows", headers=h).json()["thresholds"]


@requires_bundle
def test_copilot_turn_with_photo_is_accepted(client):
    """Regression: a new chat session was rolled back by the acceptance path (FK violation, 500)."""
    import_and_activate(fixture_bundle(), admin_id())
    h = login(client, "inspector")
    r = client.post("/v1/copilot/messages", headers=h, data={"message": "Can this casing be retreaded?"},
                    files={"file": ("t.jpg", jpeg(51), "image/jpeg")})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["inspection_id"] and body["session_id"]
    s = client.get(f"/v1/copilot/sessions/{body['session_id']}", headers=h).json()
    assert [m["role"] for m in s["messages"]] == ["user", "assistant"]
