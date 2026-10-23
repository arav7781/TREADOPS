"""Copilot turn orchestration: persist, run the LangGraph agent off-request, record the LLMOps trace."""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

from treadops_api.db import must_get, new_session, session_scope
from treadops_api.errors import ApiError
from treadops_api.models import ChatMessage, ChatSession
from treadops_api.services import inspections as insp_svc
from treadops_api.settings import get_settings
from treadops_api.storage import get_storage
from treadops_ml.common import load_yaml, sha256_bytes

from . import graph, tools
from .guardrails import config, redact
from .llm_client import LLMClient
from .observability import Tracer

log = logging.getLogger("treadops.copilot")
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="copilot")


@lru_cache
def prompt() -> tuple[str, str, str]:
    p = load_yaml(config()["prompt"])
    return p["template"], p["version"], sha256_bytes(p["template"].encode())


def llm_status() -> dict[str, Any]:
    s = get_settings()
    return {"configured": bool(s.llm_base_url and s.llm_model), "model": s.llm_model,
            "mode": "llm" if s.llm_base_url and s.llm_model else "deterministic_fallback",
            "prompt_version": prompt()[1], "prompt_sha256": prompt()[2]}


def _client() -> tuple[LLMClient | None, str | None]:
    s = get_settings()
    if not (s.llm_base_url and s.llm_model):
        return None, None
    return LLMClient(s.llm_base_url, s.llm_api_key or "not-needed"), s.llm_model


def start_turn(db, *, actor, session_id: uuid.UUID | None, text: str, image: bytes | None) -> dict[str, Any]:
    text = (text or "").strip()
    if not text and not image:
        raise ApiError(422, "empty_message", "Send a message or attach a tyre photo.")
    if session_id:
        sess = db.get(ChatSession, session_id)
        if sess is None or sess.user_id != actor.id:
            raise ApiError(404, "not_found", "Chat session not found.")
    else:
        title = (text[:60] or "Tyre photo inspection")
        sess = ChatSession(user_id=actor.id, title=redact(title)[0])
        db.add(sess)
    # Commit before acceptance: it releases its connection (rolls back the open transaction) before image work.
    db.commit()
    inspection_id = None
    assistant_id = uuid.uuid4()
    if image is not None:
        if not actor.has("inspector"):
            raise ApiError(403, "forbidden", "Uploading photos requires the inspector role.")
        insp, _ = insp_svc.accept(db, get_storage(), actor=actor, actor_id=actor.id, raw=image,
                                  idempotency_key=f"copilot:{assistant_id}", source="copilot",
                                  metadata_origin="user_provided")
        inspection_id = insp.id
        sess.inspection_id = insp.id
    user_msg = ChatMessage(session_id=sess.id, role="user", content=redact(text)[0],
                           attachments={"inspection_id": str(inspection_id)} if inspection_id else {})
    db.add(user_msg)
    asst = ChatMessage(id=assistant_id, session_id=sess.id, role="assistant", content="",
                       attachments={"status": "pending", "events": [], "inspection_id": str(inspection_id)
                                    if inspection_id else None})
    db.add(asst)
    sess.updated_at = datetime.now(UTC)
    db.commit()
    _pool.submit(_run_safely, assistant_id, actor, text, inspection_id)
    return {"session_id": str(sess.id), "user_message_id": str(user_msg.id), "assistant_message_id": str(asst.id),
            "inspection_id": str(inspection_id) if inspection_id else None, "status": "pending"}


def _emit_factory(message_id: uuid.UUID):
    def emit(kind: str, data: dict[str, Any]) -> None:
        with session_scope() as db:
            m = must_get(db, ChatMessage, message_id)
            ev = {"at": datetime.now(UTC).isoformat(), "kind": kind, **data}
            m.attachments = {**m.attachments, "events": [*m.attachments.get("events", []), ev][-40:]}
            flag_modified(m, "attachments")
    return emit


def _history(db, session_id: uuid.UUID, exclude: uuid.UUID) -> list[dict[str, Any]]:
    rows = db.execute(select(ChatMessage).where(ChatMessage.session_id == session_id, ChatMessage.id != exclude,
                                                ChatMessage.content != "")
                      .order_by(ChatMessage.created_at.desc()).limit(8)).scalars().all()
    return [{"role": m.role, "content": m.content[:2000]} for m in reversed(rows) if m.role in ("user", "assistant")]


def _run_safely(message_id: uuid.UUID, actor, text: str, inspection_id: uuid.UUID | None) -> None:
    try:
        run_turn(message_id, actor, text, inspection_id)
    except Exception as exc:
        log.exception("copilot turn failed")
        with session_scope() as db:
            m = must_get(db, ChatMessage, message_id)
            m.content = "Something went wrong while answering. The error was recorded in the LLMOps trace view."
            m.attachments = {**m.attachments, "status": "error", "error": f"{exc.__class__.__name__}"}
            flag_modified(m, "attachments")


def run_turn(message_id: uuid.UUID, actor, text: str, inspection_id: uuid.UUID | None) -> dict[str, Any]:
    cfg = config()
    system, version, sha = prompt()
    db = new_session()
    try:
        msg = must_get(db, ChatMessage, message_id)
        session_id = msg.session_id
        history = _history(db, session_id, message_id)
    finally:
        db.close()
    emit = _emit_factory(message_id)
    tracer = Tracer(session_id=session_id, user_id=actor.id, prompt_version=version, prompt_sha=sha)
    ctx = tools.ToolContext(actor=actor, inspection_id=inspection_id, emit=emit)
    llm, model = _client()
    app = graph.build(llm=llm, model=model, ctx=ctx, tracer=tracer, system_prompt=system, history=history,
                      max_iterations=int(cfg["max_agent_iterations"]), llm_params=cfg["llm"])
    emit("agent", {"message": "guardrails checked; planning" if text or inspection_id else "planning"})
    with tracer.span("copilot_turn", "node", has_attachment=inspection_id is not None):
        state = app.invoke({"user_text": text, "has_attachment": inspection_id is not None},
                           config={"recursion_limit": 40})
    blocked = bool(state.get("blocked"))
    mode = state.get("mode") or ("deterministic_fallback" if llm is None else "llm")
    status = "blocked" if blocked else ("fallback" if state.get("llm_error") else "ok")
    with session_scope() as db:
        tr = tracer.persist(db, status=status, llm_mode=mode, grounding=state.get("grounding"),
                            error=state.get("llm_error"))
        m = must_get(db, ChatMessage, message_id)
        m.content = state.get("final_text") or ""
        m.suggestions = state.get("suggestions", [])
        m.trace_id = tr.id
        m.attachments = {**m.attachments, "status": "complete", "mode": mode, "tools": tracer.tools_used,
                         "guardrails": sorted({e.rule for e in tracer.guard_events if e.action != "pass"}),
                         "grounding": state.get("grounding")}
        flag_modified(m, "attachments")
    return {"trace_id": str(tracer.trace_id), "status": status, "mode": mode}


def sweep_stale(minutes: int = 10) -> int:
    """Pending assistant messages from a crashed API process are marked failed instead of spinning forever."""
    cutoff = datetime.now(UTC) - timedelta(minutes=minutes)
    with session_scope() as db:
        rows = db.execute(select(ChatMessage).where(ChatMessage.role == "assistant", ChatMessage.content == "",
                                                    ChatMessage.created_at < cutoff)).scalars().all()
        for m in rows:
            if (m.attachments or {}).get("status") == "pending":
                m.content = "This answer was interrupted (service restart). Please ask again."
                m.attachments = {**m.attachments, "status": "error", "error": "interrupted"}
                flag_modified(m, "attachments")
        return len(rows)
