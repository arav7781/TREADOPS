"""LLMOps: copilot observability computed from stored traces, spans, guardrail events and feedback."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from treadops_ml.common import repo_path

from .. import llm_tracing
from ..db import get_db
from ..errors import ApiError
from ..models import ChatFeedback, ChatMessage, GuardrailEvent, LLMSpan, LLMTrace
from ..security import Principal, require

router = APIRouter(prefix="/v1", tags=["llmops"])


@router.get("/llmops/overview")
def overview(since: datetime | None = None, p: Principal = Depends(require("ml_engineer")),
             db: Session = Depends(get_db)):
    since = since or datetime.now(UTC) - timedelta(days=7)
    w = LLMTrace.created_at >= since
    n = db.execute(select(func.count()).where(w)).scalar_one()
    lat = db.execute(select(func.percentile_cont(0.5).within_group(LLMTrace.latency_ms),
                            func.percentile_cont(0.95).within_group(LLMTrace.latency_ms)).where(w)).one()
    tok = db.execute(select(func.coalesce(func.sum(LLMTrace.prompt_tokens), 0),
                            func.coalesce(func.sum(LLMTrace.completion_tokens), 0),
                            func.coalesce(func.sum(LLMTrace.cached_tokens), 0),
                            func.sum(LLMTrace.cost_usd), func.coalesce(func.sum(LLMTrace.n_llm_calls), 0))
                     .where(w)).one()
    by_status = dict(db.execute(select(LLMTrace.status, func.count()).where(w).group_by(LLMTrace.status)).all())
    by_mode = dict(db.execute(select(LLMTrace.llm_mode, func.count()).where(w).group_by(LLMTrace.llm_mode)).all())
    by_prompt = [{"prompt_version": v, "prompt_sha256": s[:12], "turns": c, "p95_ms": p95}
                 for v, s, c, p95 in db.execute(select(LLMTrace.prompt_version, LLMTrace.prompt_sha256, func.count(),
                                                       func.percentile_cont(0.95).within_group(LLMTrace.latency_ms))
                                                .where(w).group_by(LLMTrace.prompt_version, LLMTrace.prompt_sha256)).all()]
    tools_used = db.execute(select(func.jsonb_array_elements_text(LLMTrace.tools_used).label("t")).where(w)).scalars().all()
    tool_counts: dict[str, int] = {}
    for t in tools_used:
        tool_counts[t] = tool_counts.get(t, 0) + 1
    guards = db.execute(select(GuardrailEvent.stage, GuardrailEvent.rule, GuardrailEvent.action, func.count())
                        .join(LLMTrace, LLMTrace.id == GuardrailEvent.trace_id).where(w, GuardrailEvent.action != "pass")
                        .group_by(GuardrailEvent.stage, GuardrailEvent.rule, GuardrailEvent.action)).all()
    grounding = db.execute(select(func.count()).where(w, LLMTrace.grounding.is_not(None),
                                                      LLMTrace.grounding["pass"].astext == "true")).scalar_one()
    grounded_total = db.execute(select(func.count()).where(w, LLMTrace.grounding.is_not(None))).scalar_one()
    fb = dict(db.execute(select(ChatFeedback.score, func.count()).where(ChatFeedback.created_at >= since)
                         .group_by(ChatFeedback.score)).all())
    trunc = func.date_trunc("hour", LLMTrace.created_at)
    series = db.execute(select(trunc.label("t"), func.count(), func.percentile_cont(0.95).within_group(LLMTrace.latency_ms),
                               func.sum(case((func.jsonb_array_length(LLMTrace.guardrail_flags) > 0, 1), else_=0)))
                        .where(w).group_by("t").order_by("t")).all()
    return {
        "since": since.isoformat(), "turns": n,
        "latency_ms": {"p50": lat[0], "p95": lat[1]},
        "tokens": {"prompt": tok[0], "completion": tok[1], "cached_prompt": tok[2],
                   "cache_hit_rate": (tok[2] / tok[0]) if tok[0] else None, "llm_calls": tok[4]},
        "cost_usd": tok[3],
        "by_status": by_status, "by_mode": by_mode, "by_prompt_version": by_prompt, "tools": tool_counts,
        "guardrails": [{"stage": s, "rule": r, "action": a, "count": c} for s, r, a, c in guards],
        "grounding": {"passed": grounding, "checked": grounded_total,
                      "pass_rate": grounding / grounded_total if grounded_total else None},
        "feedback": {"up": fb.get(1, 0), "down": fb.get(-1, 0)},
        "mlflow": llm_tracing.links(),
        "timeline": [{"t": t.isoformat(), "turns": c, "p95_ms": p95, "flagged": int(f or 0)} for t, c, p95, f in series],
    }


@router.get("/llmops/traces")
def traces(status: str | None = None, rule: str | None = None, limit: int = Query(50, ge=1, le=100),
           p: Principal = Depends(require("ml_engineer")), db: Session = Depends(get_db)):
    q = select(LLMTrace)
    if status:
        q = q.where(LLMTrace.status == status)
    if rule:
        q = q.where(LLMTrace.guardrail_flags.contains([rule]))
    rows = db.execute(q.order_by(LLMTrace.created_at.desc()).limit(limit)).scalars().all()
    return {"items": [{"id": str(t.id), "created_at": t.created_at.isoformat(), "status": t.status,
                       "mode": t.llm_mode, "model": t.model, "prompt_version": t.prompt_version,
                       "latency_ms": round(t.latency_ms, 1), "tokens": t.prompt_tokens + t.completion_tokens,
                       "cost_usd": t.cost_usd, "tools": t.tools_used, "guardrail_flags": t.guardrail_flags,
                       "grounding": t.grounding, "mlflow_trace_id": t.mlflow_trace_id} for t in rows]}


@router.get("/llmops/traces/{trace_id}")
def trace(trace_id: uuid.UUID, p: Principal = Depends(require("ml_engineer")), db: Session = Depends(get_db)):
    t = db.get(LLMTrace, trace_id)
    if t is None:
        raise ApiError(404, "not_found", "Trace not found.")
    spans = db.execute(select(LLMSpan).where(LLMSpan.trace_id == t.id).order_by(LLMSpan.start_offset_ms)).scalars().all()
    events = db.execute(select(GuardrailEvent).where(GuardrailEvent.trace_id == t.id)).scalars().all()
    msg = db.execute(select(ChatMessage).where(ChatMessage.trace_id == t.id)).scalar_one_or_none()
    user_msg = None
    if msg is not None:
        user_msg = db.execute(select(ChatMessage).where(ChatMessage.session_id == msg.session_id,
                                                        ChatMessage.role == "user",
                                                        ChatMessage.created_at <= msg.created_at)
                              .order_by(ChatMessage.created_at.desc()).limit(1)).scalar_one_or_none()
    return {
        "id": str(t.id), "status": t.status, "mode": t.llm_mode, "model": t.model,
        "prompt_version": t.prompt_version, "prompt_sha256": t.prompt_sha256, "latency_ms": t.latency_ms,
        "usage": {"prompt": t.prompt_tokens, "completion": t.completion_tokens, "cached": t.cached_tokens},
        "cost_usd": t.cost_usd, "error": t.error, "grounding": t.grounding,
        "spans": [{"id": str(s.id), "parent": str(s.parent_span_id) if s.parent_span_id else None, "name": s.name,
                   "kind": s.kind, "status": s.status, "start_ms": round(s.start_offset_ms, 1),
                   "duration_ms": round(s.duration_ms, 1), "attributes": s.attributes} for s in spans],
        "guardrail_events": [{"stage": e.stage, "rule": e.rule, "action": e.action, "detail": e.detail}
                             for e in events],
        "user_message": user_msg.content if user_msg else None,
        "assistant_message": msg.content if msg else None,
        "mlflow": llm_tracing.links(t.mlflow_trace_id) if t.mlflow_trace_id else None,
    }


@router.get("/llmops/evals")
def evals(p: Principal = Depends(require("ml_engineer"))):
    d = repo_path("docs/evidence/llmops")
    reports = sorted(d.glob("copilot_eval_*.json"), reverse=True)[:10] if d.exists() else []
    return {"items": [json.loads(r.read_text()) | {"file": r.name} for r in reports]}
