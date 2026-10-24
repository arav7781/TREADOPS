"""MLflow Tracing of Bib: spans mirror the LLMOps tracer, PII and images never leave the process, and with tracing
off (or no server) a turn behaves exactly as before. The end-to-end case uses a throwaway local MLflow store."""

import types
import uuid

import pytest

from treadops_agent import observability
from treadops_api import llm_tracing

IMAGE = "data:image/jpeg;base64," + "A" * 4000


@pytest.fixture()
def fresh(monkeypatch):
    monkeypatch.setattr(llm_tracing, "_state", {"ok": None, "retry_at": 0.0, "experiment": "treadops-copilot-test",
                                                "experiment_id": None})


def _settings(uri):
    return types.SimpleNamespace(mlflow_tracking_uri=uri, llm_mlflow_tracing=True, mlflow_ui_url=None)


def _tracer():
    return observability.Tracer(session_id=uuid.uuid4(), user_id=uuid.uuid4(), prompt_version="v6", prompt_sha="ab" * 32)


def test_scrub_redacts_pii_and_drops_images():
    out = llm_tracing._scrub({"messages": [{"content": [{"type": "text", "text": "call +91 98765 43210"},
                                                        {"image_url": {"url": IMAGE}}]}], "n": 3})
    text = str(out)
    assert "98765" not in text and "[PHONE]" in text and "AAAA" not in text and "image omitted" in text
    assert out["n"] == 3


def test_tracing_off_is_a_no_op(fresh, monkeypatch):
    monkeypatch.setattr(llm_tracing, "get_settings", lambda: _settings(None))
    t = _tracer()
    with t.span("copilot_turn", "node"), t.span("guard_input", "guard") as attrs:
        attrs["x"] = 1
        t.io(inputs={"user_text": "hi"}, outputs={"blocked": False})
    assert t.mlflow_trace_id is None and [s.name for s in t.spans] == ["copilot_turn", "guard_input"]
    assert llm_tracing.links() is None


def test_a_turn_becomes_one_redacted_mlflow_trace(fresh, monkeypatch, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(llm_tracing, "get_settings", lambda: _settings(uri))
    before = mlflow.get_tracking_uri()
    try:
        t = _tracer()
        with t.span("copilot_turn", "node", has_attachment=True):
            with t.span("guard_input", "guard"):
                t.io(inputs={"user_text": "mail ada@example.com"}, outputs={"blocked": False})
            with t.span("tool:classify_tyre_image", "tool"):
                t.io(inputs={}, outputs={"triage": "manual_review", "photo": IMAGE})
            with t.span("llm_plan", "llm") as attrs:  # the graph catches LLM failures: still an error span
                attrs["error"] = "APITimeoutError: Request timed out."
            t.io(inputs={"message": "mail ada@example.com"}, outputs={"answer": "Expert check."})
            t.fail("APITimeoutError: Request timed out.")
        llm_tracing.flush()
        trace = mlflow.get_trace(t.mlflow_trace_id)
        spans = {s.name: s for s in trace.data.spans}
        assert set(spans) == {"copilot_turn", "guard_input", "tool:classify_tyre_image", "llm_plan"}
        assert spans["llm_plan"].status.status_code == "ERROR" and trace.info.state == "ERROR"
        assert spans["tool:classify_tyre_image"].status.status_code == "OK"
        assert spans["copilot_turn"].span_type == "AGENT" and spans["guard_input"].span_type == "GUARDRAIL"
        assert spans["tool:classify_tyre_image"].span_type == "TOOL"
        assert spans["guard_input"].inputs == {"user_text": "mail [EMAIL]"}
        assert "AAAA" not in str(spans["tool:classify_tyre_image"].outputs)
        assert trace.info.tags["prompt_version"] == "v6"
        assert trace.info.trace_metadata["treadops.llm_trace_id"] == str(t.trace_id)
        assert llm_tracing.links(t.mlflow_trace_id)["trace_url"].endswith(t.mlflow_trace_id)
    finally:  # undo the process-wide MLflow settings enabled() made
        mlflow.openai.autolog(disable=True)
        mlflow.config.enable_async_logging(False)
        mlflow.tracing.configure(span_processors=[])
        mlflow.set_tracking_uri(before)
