"""Bib, the TreadOps copilot, as a LangGraph state machine. Every answer is written by the LLM.

    START -> guard_input --blocked--> suggest -> END
                 |
                 v
               plan <----> tools
                 |
                 v
            guard_output -> suggest -> END        (plan --LLM error--> END, recorded as an error)

`plan` uses the OpenAI-compatible LLMClient with tool calling; the agent must fetch facts with tools. There is no
rule-based answer path: if the LLM is unavailable the turn fails visibly. Guardrails stay on both sides as safety
checks (they block or withhold, they do not write answers).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import guardrails, tools
from .observability import Tracer

# Named in homage to Bibendum ("Bib"), the Michelin Man. TreadOps is an independent prototype, not a Michelin product.
ASSISTANT_NAME = "Bib"

class AgentState(TypedDict, total=False):
    user_text: str
    has_attachment: bool
    is_video: bool
    messages: list[dict[str, Any]]
    blocked: bool
    certification_request: bool
    pending_calls: list[dict[str, Any]]
    iterations: int
    mode: str
    final_text: str
    llm_error: str | None
    suggestions: list[str]
    grounding: dict[str, Any]


FOLLOWUPS = re.compile(r"^\s*(?:\*\*)?\s*follow[- ]?ups?\s*:?\s*(?:\*\*)?\s*:?\s*$", re.I | re.M)


def split_followups(text: str) -> tuple[str, list[str]]:
    """The model ends its answer with a 'Follow-ups:' list; lift it out as the suggestion chips."""
    m = FOLLOWUPS.search(text or "")
    if not m:
        return text, []
    items = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip("*").strip()
             for line in text[m.end():].splitlines() if line.strip()]
    return text[:m.start()].rstrip(), [i for i in items if i][:4]


def verdict_label(c: dict[str, Any]) -> str:
    return {"damaged": "Damaged", "not_damaged": "Not damaged", "uncertain": "Uncertain (expert check)"}.get(
        str(c.get("verdict")), "Unknown")


NOT_A_TYRE_REPLY = ("This image does not show a tyre, so no damage verdict was produced. Please upload a photo of "
                    "the tyre: its tread, its sidewall or the whole tyre.")


def guardrail_summary(c: dict[str, Any]) -> str:
    """What the output guardrail shows when it withholds a draft that contradicts the classifier's result."""
    if c.get("verdict") == "not_a_tyre":
        return NOT_A_TYRE_REPLY
    return (f"The model's result for this tyre is **{verdict_label(c)}** ({c.get('triage_label')}, calibrated "
            f"p(defective) {c.get('p_defective', 0):.4f}). The drafted answer contradicted it, so the output "
            "guardrail withheld it. Ask again for an explanation.")


def summarize(name: str, res: dict[str, Any]) -> str:
    """One line per tool result for the chat timeline (the full result stays in the trace)."""
    if res.get("error"):
        return f"{name}: {res['error']}" + (f" - {res['message']}" if res.get("message") else "")
    if name == "classify_tyre_image" and res.get("triage"):
        return (f"{verdict_label(res)} ({res['triage_label']}) - calibrated p(defective) {res['p_defective']:.4f}, "
                f"image quality {res['quality_status']}, {res['model']['family']} v{res['model'].get('registry_version')}")
    if name == "classify_tyre_video" and res.get("triage"):
        return (f"{verdict_label(res)} - worst of {res['frames_classified']} frames is frame {res['decisive_frame'] + 1} "
                f"(p(defective) {res['p_defective']:.4f})")
    if name == "get_model_status":
        return (f"{res.get('model_family')} v{res.get('registry_version')} serving, gates {res.get('gate_status')}"
                if res.get("state") == "ready" else "no trained model available")
    if name == "get_drift_status":
        w = res.get("latest_window") or {}
        return f"latest window {w.get('status', 'none')}, {len(res.get('open_alerts', []))} open alert(s)"
    if name == "estimate_circularity_impact":
        return (f"{res['verified_retreads']} verified retread(s): {res['raw_material_saved_kg']} kg raw material, "
                f"{res['co2e_avoided_kg']} kg CO2e (estimate)")
    if name == "search_knowledge":
        return "passages: " + ", ".join(p["id"] for p in res.get("passages", [])) if res.get("passages") else "no passages"
    if name == "get_review_queue":
        return f"{res.get('awaiting_review')} inspection(s) awaiting review"
    if name == "get_preprocessing_trace":
        return f"{len(res.get('stages', []))} stages"
    if name == "list_cameras":
        return f"{res.get('count', 0)} camera(s): " + ", ".join(
            f"{c['name']} ({'live' if c.get('live') else c.get('state')})" for c in res.get("cameras", []))
    if name == "camera_quality_summary":
        return "; ".join(f"{c['camera']}: {c.get('damaged', 0)}/{c.get('inspected', 0)} damaged"
                         for c in res.get("cameras", [])) + f" (last {res.get('hours')} h)"
    if name == "inspect_camera" and res.get("triage"):
        return f"{res.get('camera')}: {verdict_label(res)} ({res['triage_label']}), p(defective) {res['p_defective']:.4f}"
    if name == "show_attribution_heatmap" and res.get("shown_in_chat"):
        return f"heatmap of inspection {str(res.get('inspection_id', ''))[:8]} shown in the chat"
    if name == "get_inspection":
        return f"inspection {str(res.get('inspection_id', ''))[:8]}: {res.get('triage_label', res.get('status'))}"
    return "done"


def session_context(inspections: list[dict[str, Any]], has_attachment: bool) -> str:
    """Earlier inspections of this conversation for the model: follow-ups refer to them without repeating the ID."""
    which = ("This message has its own attachment, so 'this tyre' means the new one; these are earlier tyres."
             if has_attachment else "'This tyre', 'this inspection' or 'the photo' means the first one listed "
             "unless the user says otherwise. Do not ask for the photo again and do not call classify_tyre_image "
             "without a new attachment.")
    return ("Inspections earlier in this conversation, newest first (stored results; quote them directly, or pass "
            "the inspection_id to get_inspection, get_preprocessing_trace or show_attribution_heatmap). "
            f"{which}\n{json.dumps(inspections, separators=(',', ':'))}")


def _normalised(raw: Any) -> str:
    try:
        value = tools.parse_arguments(raw)
    except json.JSONDecodeError:
        return "{}"  # the executor reports invalid_arguments to the model; the transcript itself stays valid
    return json.dumps(value if isinstance(value, dict) else {})


def build(*, llm, model: str | None, ctx: tools.ToolContext, tracer: Tracer, system_prompt: str,
          history: list[dict[str, Any]], max_iterations: int, llm_params: dict[str, Any]):
    def guard_input(state: AgentState) -> AgentState:
        with tracer.span("guard_input", "guard"):
            v = guardrails.check_input(state["user_text"], state["has_attachment"])
            tracer.guard_events += v.events
            tracer.io(inputs={"user_text": state["user_text"], "has_attachment": state["has_attachment"],
                              "is_video": bool(state.get("is_video"))},
                      outputs={"blocked": v.blocked, "text_for_llm": v.text, "reply": v.reply,
                               "certification_request": v.certification_request,
                               "events": [asdict(e) for e in v.events]})
        flagged = [e.rule for e in v.events if e.action != "pass"]
        ctx.emit("guard_input", {"status": "blocked" if v.blocked else "passed", "rules": flagged,
                                 "message": "input guardrails: " + ("blocked (" + ", ".join(flagged) + ")"
                                                                    if v.blocked else
                                                                    "passed" + (f" ({', '.join(flagged)})" if flagged else ""))})
        if v.blocked:
            return {"blocked": True, "final_text": v.reply or "Request blocked."}
        user_content = v.text or "Please inspect the attached tyre."
        if state.get("is_video"):
            user_content += (f"\n\n[A video of one tyre is attached: {len(ctx.frames)} sharp frames were sampled "
                             "around its circumference.]")
        elif state["has_attachment"]:
            user_content += "\n\n[A tyre photo is attached to this message.]"
        msgs = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_content}]
        if ctx.session_inspections:
            msgs.insert(1, {"role": "system", "content": session_context(ctx.session_inspections,
                                                                          state["has_attachment"])})
        if v.certification_request:
            msgs.insert(1, {"role": "system", "content": "The user is asking for a safety or certification "
                            "judgement. Explain that the model cannot certify tyres and route to an inspector."})
        return {"blocked": False, "messages": msgs, "certification_request": v.certification_request,
                "iterations": 0, "mode": "llm"}

    def plan(state: AgentState) -> AgentState:
        it = state.get("iterations", 0)
        first_tool = "classify_tyre_video" if state.get("is_video") else "classify_tyre_image"
        ctx.emit("plan", {"mode": "llm", "model": model, "iteration": it + 1,
                          "message": f"{model} is deciding the next step" + (f" (attachment: {first_tool} first)"
                                                                          if state["has_attachment"] and it == 0 else "")})
        with tracer.span("llm_plan", "llm", iteration=it, model=model) as attrs:
            try:
                force_tool = state["has_attachment"] and it == 0
                choice = {"type": "function", "function": {"name": first_tool}} if force_tool else "auto"
                for attempt in (1, 2):  # one retry: gateways occasionally return a transient 4xx/5xx
                    try:
                        msg = llm.chat_tools(model, state["messages"], tools.SCHEMAS, tool_choice=choice,
                                             temperature=llm_params["temperature"],
                                             max_tokens=llm_params["max_tokens"], timeout=llm_params["timeout_s"])
                        break
                    except Exception as exc:
                        if attempt == 2:
                            raise
                        attrs["retried_after"] = f"{exc.__class__.__name__}: {str(exc)[:200]}"
                        ctx.emit("plan", {"mode": "llm", "message": "the language model call failed; retrying once"})
                        time.sleep(0.8)
                tracer.add_usage(llm.last_usage, model)
                attrs["usage"] = llm.last_usage
                tracer.io(inputs={"iteration": it + 1, "model": model, "tool_choice": choice,
                                  "last_message": state["messages"][-1]},
                          outputs={"content": msg.get("content"), "tool_calls": msg.get("tool_calls") or [],
                                   "usage": llm.last_usage})
            except Exception as exc:
                attrs["error"] = f"{exc.__class__.__name__}: {str(exc)[:1500]}"
                filtered = "content_filter" in str(exc)
                ctx.emit("plan", {"mode": "llm", "error": attrs["error"],
                                  "message": "the model provider's content filter blocked the reply" if filtered
                                  else f"the language model did not answer ({exc.__class__.__name__})"})
                return {"llm_error": attrs["error"],
                        "final_text": ("The model provider's safety filter blocked this reply. Nothing was decided; "
                                       "please rephrase your question." if filtered else
                                       "The language model did not answer, so I have no reply to give. Nothing was "
                                       "decided; please ask again.")}
        calls = msg.get("tool_calls") or []
        clean: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
        if calls:
            # Echo arguments back as normalised JSON: providers reject malformed strings in the next request.
            clean["tool_calls"] = [{"id": c["id"], "type": "function",
                                    "function": {"name": c["function"]["name"],
                                                 "arguments": _normalised(c["function"]["arguments"])}} for c in calls]
            attrs["tool_calls"] = [c["function"]["name"] for c in calls]
        return {"messages": [*state["messages"], clean], "pending_calls": clean.get("tool_calls", []),
                "final_text": clean["content"], "iterations": it + 1}

    def run_tools(state: AgentState) -> AgentState:
        msgs = list(state["messages"])
        for call in state.get("pending_calls", []):
            name = call["function"]["name"]
            with tracer.span(f"tool:{name}", "tool", arguments=call["function"]["arguments"]) as attrs:
                ctx.emit("tool_call", {"tool": name, "args": call["function"]["arguments"]})
                res = tools.execute(ctx, name, call["function"]["arguments"])
                tracer.tools_used.append(name)
                try:
                    args = tools.parse_arguments(call["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {"unparsed": call["function"]["arguments"]}
                tracer.io(inputs=args, outputs=res)
                attrs["result_keys"] = sorted(res)[:20]
                attrs["error"] = res.get("error")
                ctx.emit("tool_result", {"tool": name, "ok": not res.get("error"), "summary": summarize(name, res)})
            msgs.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(res)[:12000]})
        return {"messages": msgs, "pending_calls": []}

    def guard_output(state: AgentState) -> AgentState:
        cls = next((r["result"] for r in ctx.results if r["tool"] in ("classify_tyre_image", "classify_tyre_video")
                    and not r["result"].get("error")), None)
        if cls is None and any(r["result"].get("error") == "not_a_tyre" for r in ctx.results
                               if r["tool"] in ("classify_tyre_image", "classify_tyre_video")):
            cls = {"verdict": "not_a_tyre"}
        body, followups = split_followups(state.get("final_text") or "")
        if not body.strip():
            ctx.emit("guard_output", {"rules": [], "message": "the language model returned no answer"})
            return {"llm_error": "empty answer", "final_text": "The language model returned an empty answer. "
                                                              "Nothing was decided; please ask again."}
        with tracer.span("guard_output", "guard"):
            v = guardrails.check_output(body, tool_results=[*ctx.session_inspections,
                                                            *[r["result"] for r in ctx.results]],
                                        system_prompt=system_prompt, classification=cls,
                                        fallback_summary=guardrail_summary(cls) if cls else None)
            tracer.guard_events += v.events
            tracer.io(inputs={"draft": body, "classification": cls},
                      outputs={"text": v.text, "replaced": v.replaced, "grounding": v.grounding,
                               "events": [asdict(e) for e in v.events]})
        flagged = [e.rule for e in v.events if e.action != "pass"]
        g = v.grounding
        ctx.emit("guard_output", {"rules": flagged, "grounding": g,
                                  "message": f"output guardrails: {g.get('checked', 0) - len(g.get('ungrounded', []))}/"
                                             f"{g.get('checked', 0)} figures grounded in tool outputs"
                                             + (f"; {', '.join(flagged)}" if flagged else "")})
        # Follow-ups are the model's own; drop them if the guardrail had to replace the answer.
        return {"final_text": v.text, "grounding": v.grounding, "suggestions": [] if v.replaced else followups}

    def suggest(state: AgentState) -> AgentState:
        return {"suggestions": state.get("suggestions", [])}

    def after_guard(state: AgentState) -> str:
        return "suggest" if state.get("blocked") else "plan"

    def after_plan(state: AgentState) -> str:
        if state.get("llm_error"):
            return "end"
        if state.get("pending_calls") and state.get("iterations", 0) <= max_iterations:
            return "tools"
        return "guard_output"

    g = StateGraph(AgentState)
    g.add_node("guard_input", guard_input)
    g.add_node("plan", plan)
    g.add_node("tools", run_tools)
    g.add_node("guard_output", guard_output)
    g.add_node("suggest", suggest)
    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", after_guard, {"plan": "plan", "suggest": "suggest"})
    g.add_conditional_edges("plan", after_plan, {"tools": "tools", "guard_output": "guard_output", "end": END})
    g.add_edge("tools", "plan")
    g.add_edge("guard_output", "suggest")
    g.add_edge("suggest", END)
    return g.compile()
