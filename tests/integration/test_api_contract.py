"""API-01..04 and DB-01 against real PostgreSQL / object storage."""

from __future__ import annotations

import json
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


def fake_llm(monkeypatch, *replies: str):
    """Bib is LLM-only. Integration tests script the model (no gateway, no credentials) through the real client."""
    from types import SimpleNamespace

    from treadops_agent import service
    from treadops_agent.llm_client import LLMClient

    script = list(replies)

    def create(**_):
        msg = SimpleNamespace(content=script.pop(0) if script else "", tool_calls=None)
        msg.model_dump = lambda: {"role": "assistant", "content": msg.content, "tool_calls": None}
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15, prompt_tokens_details=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    sdk.with_options = lambda **_: sdk
    real = service.llm_status
    monkeypatch.setattr(service, "llm_status", lambda: {**real(), "configured": True, "model": "fake-model", "mode": "llm"})
    monkeypatch.setattr(service, "_client", lambda: (LLMClient("http://fake/v1", "k", sdk_client=sdk), "fake-model"))


@requires_bundle
def test_copilot_photo_turn_and_agent_tool_creates_the_inspection(client, monkeypatch):
    """The chat turn stores only a sanitized preview; the agent's YOLO tool creates the pinned inspection."""
    from treadops_agent import service, tools

    import_and_activate(fixture_bundle(), admin_id())
    fake_llm(monkeypatch)
    started = []
    monkeypatch.setattr(service._pool, "submit", lambda fn, *args: started.append(args))  # run the agent below
    h = login(client, "inspector")
    r = client.post("/v1/copilot/messages", headers=h, data={"message": "Can this casing be retreaded?"},
                    files={"file": ("t.jpg", jpeg(51), "image/jpeg")})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["inspection_id"] is None and body["session_id"]
    s = client.get(f"/v1/copilot/sessions/{body['session_id']}", headers=h).json()
    user, assistant = s["messages"]
    assert user["attachments"]["photo"]["url"].startswith("/v1/media/") and "key" not in user["attachments"]["photo"]
    assert assistant["attachments"]["status"] == "pending" and assistant["attachments"]["has_photo"]
    viewer = login(client, "viewer")
    denied = client.post("/v1/copilot/messages", headers=viewer, data={"message": "x"},
                         files={"file": ("t.jpg", jpeg(52), "image/jpeg")})
    assert denied.status_code == 403

    (_msg_id, actor, _text, image, *_video) = started[0]
    events = []
    monkeypatch.setitem(tools.config(), "classify_wait_s", 0)
    ctx = tools.ToolContext(actor=actor, image_bytes=image, idempotency_key=f"copilot:{_msg_id}",
                            emit=lambda kind, data: events.append((kind, data)))
    first = tools.classify_tyre_image(ctx)
    again = tools.classify_tyre_image(ctx)  # the same turn never creates a second inspection
    assert ctx.inspection_id and first["inspection_id"] == again["inspection_id"] == str(ctx.inspection_id)
    assert any(k == "inspection" for k, _ in events)
    detail = client.get(f"/v1/inspections/{ctx.inspection_id}", headers=h).json()
    assert detail["source"] == "copilot" and detail["deployment"]["generation"] >= 1


def test_copilot_answer_streams_its_steps_and_final_message(client, monkeypatch):
    """SSE replays every recorded step in order, then `done` with the guarded answer; only the owner may listen."""
    from treadops_agent import service

    monkeypatch.setattr(service._pool, "submit", lambda fn, *args: fn(*args))  # finish the turn synchronously
    fake_llm(monkeypatch, "Fill the frame with the tread and keep the light even.\n\nFollow-ups:\n- Why does glare "
                          "matter?\n- Which angles are needed?\n- How is quality checked?")
    h = login(client, "inspector")
    r = client.post("/v1/copilot/messages", headers=h, data={"message": "How should I photograph a tyre?"})
    assert r.status_code == 202, r.text
    msg_id = r.json()["assistant_message_id"]
    stored = next(m for m in client.get(f"/v1/copilot/sessions/{r.json()['session_id']}", headers=h).json()["messages"]
                  if m["id"] == msg_id)

    res = client.get(f"/v1/copilot/messages/{msg_id}/stream", headers=h)
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/event-stream")
    frames = [f for f in res.text.split("\n\n") if f.startswith("event:")]
    kinds = [f.split("\n", 1)[0].removeprefix("event: ") for f in frames]
    assert kinds[-1] == "done" and kinds.count("step") == len(stored["attachments"]["events"]) > 3
    done = json.loads(frames[-1].split("data: ", 1)[1])
    assert done["content"] == stored["content"] and done["attachments"]["status"] == "complete"
    assert done["suggestions"] == ["Why does glare matter?", "Which angles are needed?", "How is quality checked?"]

    other = login(client, "viewer")
    assert client.get(f"/v1/copilot/messages/{msg_id}/stream", headers=other).status_code == 404


def test_copilot_refuses_without_a_language_model(client, monkeypatch):
    from treadops_agent import service

    real = service.llm_status
    monkeypatch.setattr(service, "llm_status", lambda: {**real(), "configured": False, "mode": "unavailable"})
    r = client.post("/v1/copilot/messages", headers=login(client, "inspector"), data={"message": "hello"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "llm_not_configured"


@requires_bundle
def test_copilot_video_turn_creates_one_inspection_per_frame(client, monkeypatch):
    """Frames of one tyre are stored as signed previews; the video tool creates one pinned inspection per frame."""
    from treadops_agent import service, tools

    import_and_activate(fixture_bundle(), admin_id())
    fake_llm(monkeypatch)
    started = []
    monkeypatch.setattr(service._pool, "submit", lambda fn, *args: started.append(args))
    h = login(client, "inspector")
    files = [("frames", (f"f{i}.jpg", jpeg(60 + i), "image/jpeg")) for i in range(3)]
    r = client.post("/v1/copilot/messages", headers=h, files=files,
                    data={"message": "Check this tyre", "video": json.dumps({"source": "camera", "duration_s": 9.5,
                                                                             "frames_considered": 38,
                                                                             "times": [0.4, 4.1, 8.8]})})
    assert r.status_code == 202, r.text
    user, assistant = client.get(f"/v1/copilot/sessions/{r.json()['session_id']}", headers=h).json()["messages"]
    video = user["attachments"]["video"]
    assert video["source"] == "camera" and [f["time_s"] for f in video["frames"]] == [0.4, 4.1, 8.8]
    assert all(f["url"].startswith("/v1/media/") and "key" not in f for f in video["frames"])
    assert assistant["attachments"]["has_video"]

    msg_id, actor, _text, _image, frames, meta = started[0]
    monkeypatch.setitem(tools.config(), "classify_wait_s", 0)
    ctx = tools.ToolContext(actor=actor, idempotency_key=f"copilot:{msg_id}", frames=frames,
                            frame_times=meta["times"], video_source=meta["source"])
    tools.classify_tyre_video(ctx)
    tools.classify_tyre_video(ctx)  # calling again never creates more inspections
    assert len(ctx.frame_inspections) == 3 == len(set(ctx.frame_inspections))
    detail = client.get(f"/v1/inspections/{ctx.frame_inspections[1]}", headers=h).json()
    assert detail["notes"] == "video frame 2/3 at 4.1 s (camera)" and detail["source"] == "copilot"

