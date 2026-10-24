"""Copilot chat: multimodal turns (text + optional tyre photo) answered by the guarded LangGraph agent."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from treadops_agent import service

from .. import llm_tracing
from ..db import get_db, must_get, new_session
from ..errors import ApiError
from ..models import ChatFeedback, ChatMessage, ChatSession, LLMTrace
from ..security import Principal, require, sign_media
from ..settings import get_settings

router = APIRouter(prefix="/v1", tags=["copilot"])
STREAM_POLL_SECONDS = 0.25
STREAM_MAX_SECONDS = 300


def message_view(m: ChatMessage) -> dict:
    att = dict(m.attachments or {})
    photo = att.get("photo")
    if isinstance(photo, dict) and photo.get("key"):  # signed short-lived link; storage keys never leave the API
        att["photo"] = {k: v for k, v in photo.items() if k != "key"} | {"url": sign_media(photo["key"])}
    video = att.get("video")
    if isinstance(video, dict):
        att["video"] = {**video, "frames": [{k: v for k, v in f.items() if k != "key"} | {"url": sign_media(f["key"])}
                                            for f in video.get("frames", []) if f.get("key")]}
    return {"id": str(m.id), "role": m.role, "content": m.content, "attachments": att,
            "suggestions": m.suggestions, "trace_id": str(m.trace_id) if m.trace_id else None,
            "created_at": m.created_at.isoformat()}


@router.get("/copilot/status")
def status(p: Principal = Depends(require("viewer"))):
    return service.llm_status()


@router.post("/copilot/messages", status_code=202)
def send(
    message: str = Form("", max_length=8000),
    session_id: uuid.UUID | None = Form(None),
    file: UploadFile | None = File(None),
    frames: list[UploadFile] | None = File(None, description="Video turns: sampled frames of one tyre, in order"),
    video: str | None = Form(None, max_length=4000, description='JSON: {"source": "clip"|"camera", "duration_s", '
                                                                 '"frames_considered", "times": [seconds per frame]}'),
    p: Principal = Depends(require("viewer")),
    db: Session = Depends(get_db),
):
    limit = get_settings().max_upload_bytes

    def read(f: UploadFile, what: str) -> bytes:
        data = f.file.read(limit + 1)
        if len(data) > limit:
            raise ApiError(413, "payload_too_large", f"{what} exceeds {limit} bytes.")
        return data

    image = read(file, "Image") if file is not None and file.filename else None
    frame_bytes = [read(f, f"Frame {i + 1}") for i, f in enumerate(frames or []) if f.filename]
    meta = None
    if video:
        try:
            meta = json.loads(video)
        except ValueError as exc:
            raise ApiError(422, "invalid_video_meta", "video must be a JSON object.") from exc
        if not isinstance(meta, dict) or meta.get("source", "clip") not in ("clip", "camera"):
            raise ApiError(422, "invalid_video_meta", "video.source must be clip or camera.")
    return service.start_turn(db, actor=p, session_id=session_id, text=message, image=image, frames=frame_bytes,
                              video=meta)


@router.get("/copilot/sessions")
def sessions(p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    rows = db.execute(select(ChatSession).where(ChatSession.user_id == p.id)
                      .order_by(ChatSession.updated_at.desc()).limit(30)).scalars().all()
    return {"items": [{"id": str(s.id), "title": s.title, "updated_at": s.updated_at.isoformat(),
                       "inspection_id": str(s.inspection_id) if s.inspection_id else None} for s in rows]}


@router.get("/copilot/sessions/{session_id}")
def session(session_id: uuid.UUID, p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    s = db.get(ChatSession, session_id)
    if s is None or s.user_id != p.id:
        raise ApiError(404, "not_found", "Chat session not found.")
    msgs = db.execute(select(ChatMessage).where(ChatMessage.session_id == s.id)
                      .order_by(ChatMessage.created_at)).scalars().all()
    return {"id": str(s.id), "title": s.title, "messages": [message_view(m) for m in msgs]}


def _owned_message(message_id: uuid.UUID, user_id: uuid.UUID) -> dict:
    db = new_session()
    try:
        m = db.get(ChatMessage, message_id)
        sess = db.get(ChatSession, m.session_id) if m is not None else None
        if m is None or m.role != "assistant" or sess is None or sess.user_id != user_id:
            raise ApiError(404, "not_found", "Assistant message not found.")
        return message_view(m)
    finally:
        db.close()


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.get("/copilot/messages/{message_id}/stream", response_class=StreamingResponse,
            responses={200: {"content": {"text/event-stream": {}}}})
async def stream(message_id: uuid.UUID, request: Request, p: Principal = Depends(require("viewer"))):
    """Server-sent events for one answer: `step` for each agent/pipeline event as it is recorded, `inspection` once
    the inspection tool has created one, and `done` with the final, guardrail-checked message. The answer text is only
    sent after the output guardrails have seen all of it."""
    first = await run_in_threadpool(_owned_message, message_id, p.id)

    async def events() -> AsyncIterator[str]:
        sent, inspection, msg = 0, None, first
        deadline = time.monotonic() + STREAM_MAX_SECONDS
        last_beat = time.monotonic()
        while True:
            att = msg["attachments"]
            steps = att.get("events") or []
            for ev in steps[min(sent, len(steps)):]:  # the stored list is capped, so never index past its end
                yield _sse("step", ev)
            sent = max(sent, len(steps))
            if att.get("inspection_id") and att["inspection_id"] != inspection:
                inspection = att["inspection_id"]
                yield _sse("inspection", {"inspection_id": inspection})
            if att.get("status") != "pending":
                yield _sse("done", msg)
                return
            if time.monotonic() > deadline or await request.is_disconnected():
                yield _sse("timeout", {"message": "Stream closed; the answer will still appear on reload."})
                return
            if time.monotonic() - last_beat > 10:
                last_beat = time.monotonic()
                yield ": keep-alive\n\n"
            await asyncio.sleep(STREAM_POLL_SECONDS)
            msg = await run_in_threadpool(_owned_message, message_id, p.id)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


class FeedbackIn(BaseModel):
    score: int = Field(ge=-1, le=1)
    comment: str | None = Field(None, max_length=1000)


@router.post("/copilot/messages/{message_id}/feedback")
def feedback(message_id: uuid.UUID, body: FeedbackIn, p: Principal = Depends(require("viewer")),
             db: Session = Depends(get_db)):
    if body.score == 0:
        raise ApiError(422, "invalid_score", "score must be -1 or 1")
    m = db.get(ChatMessage, message_id)
    if m is None or m.role != "assistant":
        raise ApiError(404, "not_found", "Assistant message not found.")
    sess = must_get(db, ChatSession, m.session_id)
    if sess.user_id != p.id:
        raise ApiError(403, "forbidden", "You can only rate answers in your own sessions.")
    stmt = insert(ChatFeedback).values(message_id=m.id, user_id=p.id, score=body.score, comment=body.comment)
    db.execute(stmt.on_conflict_do_update(constraint="uq_feedback_once",
                                          set_={"score": body.score, "comment": body.comment}))
    db.commit()
    trace = db.get(LLMTrace, m.trace_id) if m.trace_id else None
    if trace is not None and trace.mlflow_trace_id:  # the same rating on the MLflow trace (best effort)
        llm_tracing.log_assessment(trace.mlflow_trace_id, "user_feedback", body.score > 0, human=str(p.id),
                                   rationale=body.comment)
    return {"message_id": str(m.id), "score": body.score}
