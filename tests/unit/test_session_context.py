"""Follow-up turns: earlier inspections reach the model and their figures count as grounded; IDs are not PII."""

import json

from treadops_agent import graph, guardrails, tools
from treadops_agent.llm_client import LLMClient
from treadops_agent.observability import Tracer

from .test_agent_llm_path import CLASSIFIED, FakeSDK, _Msg

EARLIER = {**CLASSIFIED, "inspection_id": "84e4dfd9-b218-42e8-9123-456789f155b2", "p_defective": 0.8832,
           "raw_probs": {"defective": 0.9599, "good": 0.0401}}


def test_a_follow_up_sees_the_earlier_inspection_and_its_figures_are_grounded():
    sdk = FakeSDK([_Msg(content="Inspection 84e4dfd9-b218-42e8-9123-456789f155b2 was Damaged with a calibrated "
                                "p(defective) of 88.3% (raw 95.99%).")])
    ctx = tools.ToolContext(actor=None, session_inspections=[EARLIER])
    app = graph.build(llm=LLMClient("http://fake/v1", "k", sdk_client=sdk), model="fake", ctx=ctx,
                      tracer=Tracer(session_id=None, user_id=None, prompt_version="v1", prompt_sha="x" * 64),
                      system_prompt="You are TreadOps Copilot. " * 10, history=[], max_iterations=3,
                      llm_params={"temperature": 0.2, "max_tokens": 300, "timeout_s": 5})
    state = app.invoke({"user_text": "Can you show me the occlusion heatmap for this inspection?",
                        "has_attachment": False})
    context = [m["content"] for m in sdk.requests[0]["messages"] if m["role"] == "system"][1]
    assert "show_attribution_heatmap" in context and "Do not ask for the photo again" in context
    assert json.loads(context.split("\n", 1)[1])[0]["inspection_id"] == EARLIER["inspection_id"]
    assert state["grounding"]["pass"] and not state["grounding"]["ungrounded"]
    assert "84e4dfd9-b218-42e8-9123-456789f155b2" in state["final_text"]  # not redacted as a phone number


def test_a_negative_tool_value_quoted_with_its_sign_is_grounded():
    tool = {"min_delta": -0.504, "max_delta": 2.095}
    v = guardrails.check_output("Max delta +2.095; min delta -0.504 (a mild blue region).", tool_results=[tool],
                                system_prompt="x" * 200, classification=None, fallback_summary=None)
    assert v.grounding["pass"] and v.grounding["checked"] == 1  # 2.095 is not a fraction or percentage


def test_ids_and_hashes_are_not_pii_but_phones_and_cards_are():
    text, counts = guardrails.redact("inspection 84e4dfd9-b218-42e8-9123-456789f155b2, sha 5ec3b6a3c0ffee1234567890ab, "
                                     "call +91 98765 43210, card 4111 1111 1111 1111")
    assert "84e4dfd9-b218-42e8-9123-456789f155b2" in text and "5ec3b6a3c0ffee1234567890ab" in text
    assert "98765" not in text and "4111" not in text
    assert counts == {"phone": 1, "card_number": 1}


def test_each_pipeline_stage_reaches_the_chat_once_even_when_depth_lands_before_quality(monkeypatch):
    import uuid

    def st(i: int) -> dict:
        return {"index": i, "name": f"s{i}", "title": f"Stage {i}", "duration_ms": 1}

    polls = [[st(1), st(2)], [st(1), st(2), st(9)], [st(1), st(2), st(5), st(9)]]  # listed in index order

    def load(db, iid):
        stages = polls.pop(0)
        return {"inspection_id": str(iid), "status": "running" if polls else "succeeded", "stages": stages}

    class _DB:
        def close(self): ...

    emitted: list[int] = []
    monkeypatch.setattr(tools, "_load", load)
    monkeypatch.setattr(tools, "new_session", _DB)
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)
    ctx = tools.ToolContext(actor=None, inspection_id=uuid.uuid4(),
                            emit=lambda kind, ev: emitted.append(ev["stage"]) if kind == "stage" else None)
    assert tools.classify_tyre_image(ctx)["status"] == "succeeded"
    assert emitted == [1, 2, 9, 5]
