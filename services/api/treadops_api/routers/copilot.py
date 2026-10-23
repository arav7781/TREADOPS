"""Copilot chat: multimodal turns (text + optional tyre photo) answered by the guarded LangGraph agent."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from treadops_agent import service

from ..db import get_db, must_get
from ..errors import ApiError
from ..models import ChatFeedback, ChatMessage, ChatSession
from ..security import Principal, require, sign_media
from ..settings import get_settings

router = APIRouter(prefix="/v1", tags=["copilot"])


def message_view(m: ChatMessage) -> dict:
    att = dict(m.attachments or {})
    photo = att.get("photo")
    if isinstance(photo, dict) and photo.get("key"):  # signed short-lived link; storage keys never leave the API
        att["photo"] = {k: v for k, v in photo.items() if k != "key"} | {"url": sign_media(photo["key"])}
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
    p: Principal = Depends(require("viewer")),
    db: Session = Depends(get_db),
):
    image = None
    if file is not None and file.filename:
        limit = get_settings().max_upload_bytes
        image = file.file.read(limit + 1)
        if len(image) > limit:
            raise ApiError(413, "payload_too_large", f"Image exceeds {limit} bytes.")
    return service.start_turn(db, actor=p, session_id=session_id, text=message, image=image)


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
    return {"message_id": str(m.id), "score": body.score}
