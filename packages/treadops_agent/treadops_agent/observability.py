"""LLMOps tracing: one trace per copilot turn, spans for graph nodes, LLM calls and tool calls."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from treadops_api.models import GuardrailEvent, LLMSpan, LLMTrace

from .guardrails import GuardEvent, config


@dataclass
class SpanRecord:
    id: uuid.UUID
    parent: uuid.UUID | None
    name: str
    kind: str
    start_ms: float
    duration_ms: float = 0.0
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)


class Tracer:
    def __init__(self, *, session_id, user_id, prompt_version: str, prompt_sha: str):
        self.trace_id = uuid.uuid4()
        self.session_id, self.user_id = session_id, user_id
        self.prompt_version, self.prompt_sha = prompt_version, prompt_sha
        self._t0 = time.perf_counter()
        self.spans: list[SpanRecord] = []
        self.guard_events: list[GuardEvent] = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0}
        self.n_llm = 0
        self.tools_used: list[str] = []
        self.model: str | None = None
        self._stack: list[uuid.UUID] = []

    def _now(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    @contextmanager
    def span(self, name: str, kind: str, **attributes):
        rec = SpanRecord(uuid.uuid4(), self._stack[-1] if self._stack else None, name, kind, self._now(),
                         attributes=dict(attributes))
        self.spans.append(rec)
        self._stack.append(rec.id)
        try:
            yield rec.attributes
        except Exception as exc:
            rec.status = "error"
            rec.attributes["error"] = f"{exc.__class__.__name__}: {str(exc)[:300]}"
            raise
        finally:
            rec.duration_ms = self._now() - rec.start_ms
            self._stack.pop()

    def add_usage(self, usage: dict[str, int] | None, model: str) -> None:
        self.n_llm += 1
        self.model = model
        if usage:
            for k in self.usage:
                self.usage[k] += int(usage.get(k, 0))

    def cost_usd(self) -> float | None:
        price = config().get("pricing_per_million_tokens", {}).get(self.model or "")
        if not price:
            return None
        return round((self.usage["prompt_tokens"] * price["input"] + self.usage["completion_tokens"] * price["output"])
                     / 1e6, 6)

    def persist(self, db, *, status: str, llm_mode: str, grounding: dict | None, error: str | None = None) -> LLMTrace:
        tr = LLMTrace(
            id=self.trace_id, session_id=self.session_id, user_id=self.user_id, status=status, llm_mode=llm_mode,
            model=self.model, prompt_version=self.prompt_version, prompt_sha256=self.prompt_sha,
            latency_ms=self._now(), prompt_tokens=self.usage["prompt_tokens"],
            completion_tokens=self.usage["completion_tokens"], cached_tokens=self.usage["cached_prompt_tokens"],
            cost_usd=self.cost_usd(), n_llm_calls=self.n_llm, n_tool_calls=len(self.tools_used),
            tools_used=self.tools_used, guardrail_flags=sorted({e.rule for e in self.guard_events
                                                                 if e.action != "pass"}),
            grounding=grounding, error=error,
        )
        db.add(tr)
        db.flush()
        for s in self.spans:
            db.add(LLMSpan(id=s.id, trace_id=tr.id, parent_span_id=s.parent, name=s.name, kind=s.kind, status=s.status,
                           start_offset_ms=s.start_ms, duration_ms=s.duration_ms, attributes=s.attributes))
        for e in self.guard_events:
            db.add(GuardrailEvent(trace_id=tr.id, stage=e.stage, rule=e.rule, action=e.action, detail=e.detail))
        return tr
