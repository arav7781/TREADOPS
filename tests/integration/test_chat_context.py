"""Follow-up chat turns see the conversation's earlier inspections, and the heatmap tool shows a stored one.

Seen in chat: after a photo turn, "show me the occlusion heatmap for this inspection" got "the original photo is
no longer in this thread", because the next turn only had the answer text (with the ID redacted).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from treadops_api import db as dbmod
from treadops_api.models import ChatMessage, ChatSession

from .conftest import admin_id, fixture_bundle, import_and_activate, jpeg, login, requires_bundle


def _inspect(client, key: str, seed: int) -> str:
    from treadops_worker.tasks import run_job

    acc = client.post("/v1/inspections", headers={**login(client, "inspector"), "Idempotency-Key": key},
                      files={"file": ("t.jpg", jpeg(seed), "image/jpeg")}).json()
    run_job(acc["job_id"])
    return acc["inspection_id"]


@requires_bundle
def test_follow_up_turns_get_the_sessions_inspections_and_can_show_the_heatmap(client):
    from treadops_agent import service, tools

    import_and_activate(fixture_bundle(), admin_id())
    photo, looked_up = _inspect(client, "ctx-photo", 91), _inspect(client, "ctx-ref", 92)
    t0 = datetime.now(UTC) - timedelta(minutes=5)
    with dbmod.session_scope() as db:
        sess = ChatSession(user_id=admin_id(), title="Tyre photo inspection")
        db.add(sess)
        db.flush()
        db.add_all([
            ChatMessage(session_id=sess.id, role="assistant", content="Damaged.", created_at=t0,
                        attachments={"status": "complete", "inspection_id": photo}),
            ChatMessage(session_id=sess.id, role="assistant", content="Here it is.", created_at=t0 + timedelta(minutes=1),
                        attachments={"status": "complete", "inspection_refs": [{"id": looked_up, "focus": None}]}),
        ])
        pending = ChatMessage(session_id=sess.id, role="assistant", content="", attachments={"status": "pending"})
        db.add(pending)
        db.flush()
        session_id, pending_id = sess.id, pending.id

    with dbmod.session_scope() as db:
        earlier = service._session_inspections(db, session_id, pending_id)
    assert [i["inspection_id"] for i in earlier] == [looked_up, photo]  # newest first
    assert all("stages" not in i and i["p_defective"] is not None for i in earlier)

    events: list[tuple[str, dict]] = []
    ctx = tools.ToolContext(actor=None, emit=lambda k, d: events.append((k, d)), session_inspections=earlier)
    missing = tools.classify_tyre_image(ctx)  # no new photo: point the model at the earlier inspections
    assert missing["error"] == "no_attachment" and missing["earlier_inspections"] == [looked_up, photo]

    heat = tools.show_attribution_heatmap(ctx, photo)
    assert heat["shown_in_chat"] and heat["grid"] == 8 and 1 <= heat["strongest_cell"]["row"] <= 8
    assert ("inspection_ref", {"inspection_id": photo, "focus": "attribution",
                               "message": f"showing inspection {photo[:8]} at its attribution stage"}) in events
    assert tools.show_attribution_heatmap(ctx, str(uuid.uuid4()))["error"] == "no_heatmap"

    emit = service._emit_factory(pending_id)  # the answer records each shown inspection once, latest focus
    emit("inspection_ref", {"inspection_id": photo, "focus": "classified"})
    emit("inspection_ref", {"inspection_id": photo, "focus": "attribution"})
    with dbmod.session_scope() as db:
        assert db.get(ChatMessage, pending_id).attachments["inspection_refs"] == [{"id": photo, "focus": "attribution"}]
