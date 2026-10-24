"""LLMOps tracing: one trace per copilot turn, spans for graph nodes, LLM calls and tool calls.

Stored in PostgreSQL (the LLMOps page) and mirrored, span for span, into MLflow Tracing when it is on
(treadops_api.llm_tracing): the same tree, with full inputs and outputs, in the MLflow UI."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from treadops_api import llm_tracing
from treadops_api.models import GuardrailEvent, LLMSpan, LLMTrace

from .guardrails import GuardEvent, config, redact

llm_tracing.register_redactor(lambda text: redact(text)[0])


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
        self._ml: list[Any] = []  # open MLflow spans (None entries when tracing is off)
        self.mlflow_trace_id: str | None = None

    def _now(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    @contextmanager
    def span(self, name: str, kind: str, **attributes):
        root = not self._stack
        rec = SpanRecord(uuid.uuid4(), self._stack[-1] if self._stack else None, name, kind, self._now(),
                         attributes=dict(attributes))
        self.spans.append(rec)
        self._stack.append(rec.id)
        with llm_tracing.span(name, "turn" if root else kind, {"treadops.span_id": str(rec.id), **attributes}) as ml:
            self._ml.append(ml)
            if root and ml is not None:
                self.mlflow_trace_id = ml.trace_id
                llm_tracing.tag_trace(
                    session_id=str(self.session_id), user=str(self.user_id),
                    tags={"prompt_version": self.prompt_version, "app": "treadops-bib"},
                    metadata={"treadops.llm_trace_id": str(self.trace_id), "prompt_sha256": self.prompt_sha})
            try:
                yield rec.attributes
            except Exception as exc:
                rec.status = "error"
                rec.attributes["error"] = f"{exc.__class__.__name__}: {str(exc)[:300]}"
                raise
            finally:
                rec.duration_ms = self._now() - rec.start_ms
                self._stack.pop()
                self._ml.pop()
                llm_tracing.set_io(ml, **{f"treadops.{k}": v for k, v in rec.attributes.items()
                                          if isinstance(v, str | int | float | bool)})
                if kind == "llm" and rec.attributes.get("error"):  # caught by the graph, so not raised
                    llm_tracing.mark_error(ml, rec.attributes["error"])

    def fail(self, message: str) -> None:
        """Mark the innermost open span (the turn, at the end of run_turn) as failed in MLflow."""
        if self._ml:
            llm_tracing.mark_error(self._ml[-1], message)

    def io(self, inputs: Any = None, outputs: Any = None) -> None:
        """Full inputs/outputs of the innermost open span, for MLflow (PostgreSQL keeps the compact attributes)."""
        if self._ml:
            llm_tracing.set_io(self._ml[-1], inputs=inputs, outputs=outputs)

    def add_usage(self, usage: dict[str, int] | None, model: str | None) -> None:
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
            grounding=grounding, error=error, mlflow_trace_id=self.mlflow_trace_id,
        )
        db.add(tr)
        db.flush()
        for s in self.spans:
            db.add(LLMSpan(id=s.id, trace_id=tr.id, parent_span_id=s.parent, name=s.name, kind=s.kind, status=s.status,
                           start_offset_ms=s.start_ms, duration_ms=s.duration_ms, attributes=s.attributes))
        for e in self.guard_events:
            db.add(GuardrailEvent(trace_id=tr.id, stage=e.stage, rule=e.rule, action=e.action, detail=e.detail))
        return tr
