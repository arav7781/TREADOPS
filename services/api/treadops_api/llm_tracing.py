"""MLflow Tracing for TreadOps' LLM calls: Bib's turns (API) and the vision second opinion (inference worker).

A Bib turn is one MLflow trace in experiment `treadops-copilot`, mirroring the LLMOps spans kept in PostgreSQL
(observability.Tracer): turn -> guard_input -> llm_plan -> tool:<name> ... -> guard_output. The OpenAI SDK is
autologged, so every chat completion appears inside its llm_plan span with the chat view and token usage. The
turn carries session, user, prompt version/hash and the LLMOps trace ID, and inspectors' thumbs up/down are
logged on it as feedback. The vision second opinion goes to `treadops-vision-check`, one trace per check.

Before export every span passes `_scrub`: the registered redactors (the copilot's PII redaction) run on every
string, and base64 images are replaced by a size note. Tracing is best effort: it is off unless
MLFLOW_TRACKING_URI is set (LLM_MLFLOW_TRACING=false turns it off), and an unreachable server never fails a turn.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from .settings import get_settings

log = logging.getLogger("treadops.llm_tracing")

COPILOT_EXPERIMENT = "treadops-copilot"
VISION_EXPERIMENT = "treadops-vision-check"
RETRY_S = 300  # after a failed setup (server down), try again this much later
_DATA_URL = re.compile(r"data:image/[\w.+-]+;base64,[A-Za-z0-9+/=]+")
_lock = threading.Lock()
_state: dict[str, Any] = {"ok": None, "retry_at": 0.0, "experiment": COPILOT_EXPERIMENT, "experiment_id": None}
_redactors: list[Callable[[str], str]] = []


def register_redactor(fn: Callable[[str], str]) -> None:
    if fn not in _redactors:
        _redactors.append(fn)


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        value = _DATA_URL.sub(lambda m: f"[image omitted: {len(m.group(0)) * 3 // 4096} KB base64]", value)
        for fn in _redactors:
            value = fn(value)
        return value
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_scrub(v) for v in value]
    return value


def _scrub_span(span) -> None:
    """Span processor: runs on every span (ours and autologged) before it leaves the process."""
    if span.inputs is not None:
        span.set_inputs(_scrub(span.inputs))
    if span.outputs is not None:
        span.set_outputs(_scrub(span.outputs))


def use_experiment(name: str) -> None:
    """Which experiment this process traces into (set before the first span; the API uses the copilot's)."""
    _state["experiment"] = name


def enabled() -> bool:
    """Set MLflow tracing up once per process (lazily, on first use). False when off or the server is unreachable."""
    s = get_settings()
    if not s.mlflow_tracking_uri or not s.llm_mlflow_tracing:
        return False
    if _state["ok"] or time.time() < _state["retry_at"]:
        return bool(_state["ok"])
    with _lock:
        if _state["ok"] or time.time() < _state["retry_at"]:
            return bool(_state["ok"])
        try:
            import mlflow

            mlflow.set_tracking_uri(s.mlflow_tracking_uri)
            experiment = _state["experiment"]
            _state["experiment_id"] = mlflow.set_experiment(experiment).experiment_id
            mlflow.tracing.configure(span_processors=[_scrub_span])
            mlflow.config.enable_async_logging(True)  # export in the background; a turn never waits on MLflow
            mlflow.openai.autolog()
            _state["ok"] = True
            log.info("MLflow tracing on: experiment %s (%s)", experiment, _state["experiment_id"])
        except Exception as exc:  # server down or misconfigured: chat keeps working, tracing retries later
            _state.update(ok=False, retry_at=time.time() + RETRY_S)
            log.warning("MLflow tracing unavailable (%s: %s); retrying in %ss", exc.__class__.__name__, exc, RETRY_S)
    return bool(_state["ok"])


def experiment_id() -> str | None:
    return _state["experiment_id"] if enabled() else None


def links(trace_id: str | None = None) -> dict[str, str] | None:
    """Browser links into the MLflow UI: this process's experiment's traces, and one trace when given."""
    exp = experiment_id()
    s = get_settings()
    base = (s.mlflow_ui_url or s.mlflow_tracking_uri or "").rstrip("/")
    if exp is None or not base:
        return None
    out = {"experiment": _state["experiment"], "traces_url": f"{base}/#/experiments/{exp}/traces"}
    if trace_id:
        out["trace_id"] = trace_id
        out["trace_url"] = f"{base}/#/experiments/{exp}/traces?selectedEvaluationId={trace_id}"
    return out


SPAN_TYPES = {"turn": "AGENT", "node": "CHAIN", "guard": "GUARDRAIL", "llm": "CHAIN", "tool": "TOOL",
              "vision": "AGENT"}


@contextmanager
def span(name: str, kind: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """An MLflow span when tracing is on (yields the LiveSpan), else a no-op (yields None). Never raises from MLflow."""
    if not enabled():
        yield None
        return
    import mlflow
    from mlflow.entities import SpanType

    try:
        cm = mlflow.start_span(name=name, span_type=getattr(SpanType, SPAN_TYPES.get(kind, "UNKNOWN")),
                               attributes={k: v for k, v in (attributes or {}).items() if v is not None})
        live = cm.__enter__()
    except Exception as exc:
        log.warning("could not open MLflow span %s: %s", name, exc)
        yield None
        return
    try:
        yield live
    except BaseException as exc:
        cm.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        cm.__exit__(None, None, None)


def set_io(live, inputs: Any = None, outputs: Any = None, **attributes) -> None:
    if live is None:
        return
    try:
        if inputs is not None:
            live.set_inputs(inputs)
        if outputs is not None:
            live.set_outputs(outputs)
        if attributes:
            live.set_attributes({k: v for k, v in attributes.items() if v is not None})
    except Exception as exc:
        log.debug("span update failed: %s", exc)


def mark_error(live, message: str) -> None:
    """An error the code handled (the turn still answers) is still an error in the trace view."""
    if live is None:
        return
    try:
        from mlflow.entities import SpanStatus, SpanStatusCode

        live.set_status(SpanStatus(SpanStatusCode.ERROR, str(message)[:500]))
    except Exception as exc:
        log.debug("span status update failed: %s", exc)


def tag_trace(*, session_id: str | None = None, user: str | None = None, tags: dict[str, str] | None = None,
              metadata: dict[str, str] | None = None) -> None:
    """Session, user, tags and metadata on the trace that is open in this thread."""
    if not _state["ok"]:
        return
    try:
        import mlflow

        mlflow.update_current_trace(session_id=session_id, user=user, tags=tags, metadata=metadata)
    except Exception as exc:
        log.debug("trace update failed: %s", exc)


def log_assessment(trace_id: str | None, name: str, value: Any, *, human: str | None = None,
                   rationale: str | None = None) -> None:
    """A feedback value on a finished trace: a person's (thumbs up/down) or code's (grounding rate)."""
    if not trace_id or not enabled():
        return
    try:
        import mlflow
        from mlflow.entities import AssessmentSource, AssessmentSourceType

        source = (AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=human) if human else
                  AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="treadops.guardrails"))
        mlflow.log_feedback(trace_id=trace_id, name=name, value=value, source=source,
                            rationale=_scrub(rationale) if rationale else None)
    except Exception as exc:
        log.warning("could not log %s on trace %s: %s", name, trace_id, exc)


def flush() -> None:
    if _state["ok"]:
        try:
            import mlflow

            mlflow.flush_trace_async_logging()
        except Exception as exc:
            log.debug("flush failed: %s", exc)
