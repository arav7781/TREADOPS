"""Vision second opinion: answer parsing, the escalation rule, the three-way verdict and the guardrail."""

import pytest

from treadops_agent import guardrails as g
from treadops_agent.tools import aggregate_frames, verdict_of
from treadops_worker.vision_check import decide, parse


def test_answers_are_parsed_or_marked_unusable():
    ok = parse('Sure: {"damage_visible": true, "confidence": "High", "findings": ["gash", "exposed cords"], '
               '"location": "sidewall"}')
    assert ok == {"tyre_visible": None, "subject": None, "damage_visible": True, "confidence": "high",
                  "findings": ["gash", "exposed cords"], "location": "sidewall"}
    logo = parse('{"tyre_visible": false, "subject": "a company logo", "damage_visible": false, "findings": []}')
    assert logo["tyre_visible"] is False and logo["subject"] == "a company logo"
    assert parse("I cannot tell from this photo.")["error"] == "unparseable_answer"
    assert parse('{"damage_visible": "maybe"}')["error"] == "unparseable_answer"


@pytest.mark.parametrize(("policy", "opinion", "expected"), [
    ("defect_flagged", {"damage_visible": False, "confidence": "high"}, ("defect_flagged", "not_needed")),
    ("manual_review", None, ("manual_review", "not_needed")),
    ("good_candidate", {"damage_visible": True, "confidence": "high", "findings": ["gash"]}, ("manual_review", "disagrees")),
    ("good_candidate", {"damage_visible": True, "confidence": "medium"}, ("manual_review", "disagrees")),
    ("good_candidate", {"damage_visible": True, "confidence": "low"}, ("good_candidate", "agrees")),
    ("good_candidate", {"damage_visible": False, "confidence": "high"}, ("good_candidate", "agrees")),
    ("good_candidate", {"error": "Timeout"}, ("manual_review", "unavailable")),
])
def test_the_second_opinion_can_only_send_tyres_to_an_expert(policy, opinion, expected):
    triage, decision, reason = decide(policy, opinion)
    assert (triage, decision) == expected
    assert (reason is not None) == (decision in ("disagrees", "unavailable"))


def test_not_damaged_only_for_a_clean_auto_clear():
    assert verdict_of(True, "manual_review", "accepted") == "damaged"
    assert verdict_of(False, "good_candidate", "accepted") == "not_damaged"
    assert verdict_of(False, "manual_review", "accepted") == "uncertain"      # e.g. escalated by the second opinion
    assert verdict_of(False, "good_candidate", "low_quality") == "uncertain"


def test_an_uncertain_frame_makes_an_otherwise_clean_tyre_uncertain():
    f = lambda i, v, t, p: {"frame": i, "time_s": float(i), "inspection_id": str(i), "status": "succeeded",  # noqa: E731
                            "verdict": v, "triage": t, "p_defective": p, "quality_status": "accepted"}
    out = aggregate_frames([f(0, "not_damaged", "good_candidate", 0.02), f(1, "uncertain", "manual_review", 0.12)])
    assert out["verdict"] == "uncertain" and out["decisive_frame"] == 1


def test_an_uncertain_result_may_be_called_possibly_damaged_but_never_not_damaged():
    cls = {"triage": "manual_review", "verdict": "uncertain", "p_defective": 0.12}
    no = g.check_output("Good news: the tyre is not damaged.", tool_results=[cls], system_prompt="x" * 200,
                        classification=cls, fallback_summary="withheld")
    assert no.replaced
    ok = g.check_output("Uncertain (expert check): it may be damaged; the second opinion saw a gash. Expert review.",
                        tool_results=[cls], system_prompt="x" * 200, classification=cls, fallback_summary="withheld")
    assert not ok.replaced
    # Seen in chat: explaining the disagreement quotes YOLO's own view; that is not a contradiction.
    quoted = g.check_output("YOLO alone scored it not damaged, but the vision check sees a gash, so the models disagree "
                            "and it goes to Expert review.", tool_results=[cls], system_prompt="x" * 200,
                            classification=cls, fallback_summary="withheld")
    assert not quoted.replaced


def test_a_rejected_non_tyre_gets_no_verdict_from_the_answer():
    cls = {"verdict": "not_a_tyre"}
    ok = g.check_output("This image shows a logo, not a tyre, so no damage verdict was produced. Please upload a "
                        "photo of the tyre.", tool_results=[], system_prompt="x" * 200, classification=cls,
                        fallback_summary="withheld")
    assert not ok.replaced
    for draft in ("The tyre is not damaged.", "Verdict: Damaged.", "It is a retread candidate."):
        bad = g.check_output(draft, tool_results=[], system_prompt="x" * 200, classification=cls,
                             fallback_summary="withheld")
        assert bad.replaced and any(e.rule == "verdict_on_non_tyre" for e in bad.events)


def test_a_video_whose_frames_are_all_rejected_is_not_a_tyre():
    frames = [{"frame": i, "time_s": float(i), "inspection_id": str(i), "status": "rejected"} for i in range(3)]
    out = aggregate_frames(frames)
    assert out["error"] == "not_a_tyre" and out["frames_rejected"] == 3
