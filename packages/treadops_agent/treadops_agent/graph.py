"""TreadOps Copilot as a LangGraph state machine.

    START -> guard_input --blocked--> finalize
                 |                        ^
                 v                        |
               plan <----> tools          |
                 | (llm error / no llm)   |
                 v                        |
              fallback ----------> guard_output -> suggest -> END

`plan` uses the OpenAI-compatible LLMClient with tool calling. `fallback` is a deterministic, tool-grounded
planner used when no LLM is configured or a call fails; its trace is labelled `deterministic_fallback`.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import guardrails, tools
from .observability import Tracer

# Named in homage to Bibendum ("Bib"), the Michelin Man. TreadOps is an independent prototype, not a Michelin product.
ASSISTANT_NAME = "Bib"

STEP_LABEL = {"good_candidate": ["Route the casing to retread intake for the retreader's full casing inspection.",
                                 "Record your verified disposition so the impact ledger can count it.",
                                 "Capture the opposite sidewall and bead if they were not photographed."],
              "manual_review": ["Inspect the casing in person, focusing on the regions highlighted in red.",
                                "Capture additional angles (sidewall, shoulder, crown) in even light.",
                                "Record good / defective / unjudgeable with a reason in the review queue."],
              "defect_flagged": ["Hold the casing out of the retread flow and confirm the defect visually.",
                                 "If confirmed, record the disposition as material recovery.",
                                 "If you disagree, label it with a reason; disagreements feed model monitoring."]}


class AgentState(TypedDict, total=False):
    user_text: str
    has_attachment: bool
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


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def compose_classification(c: dict[str, Any]) -> str:
    if c.get("error"):
        msgs = {"no_attachment": "Attach a tyre photo and I'll run the deployed YOLO classifier on it.",
                "timeout": "The inspection is still in the queue; open it from the inspection page when it completes.",
                "inspection_failed": "The inspection job failed; see the inspection page for the typed error."}
        return msgs.get(c["error"], f"Classification unavailable ({c['error']}).")
    rp = c.get("recommended_pathway") or {}
    lines = [
        f"**Triage: {c['triage_label']}** - calibrated p(defective) {c['p_defective']:.4f} "
        f"(raw softmax {c['raw_probs'].get('defective', 0):.4f}) from {c['model']['family']} "
        f"v{c['model'].get('registry_version') or '?'} (bundle `{c['model']['bundle_sha256']}`).",
        f"Why: {c['triage_reason']}.",
        f"Image quality: **{c['quality_status']}**" + (f" ({', '.join(c['quality_reasons'])})"
                                                        if c.get("quality_reasons") else "") + ".",
    ]
    if rp.get("label"):
        lines.append(f"Recommended pathway: **{rp['label']}** - {rp.get('action')}")
    est = rp.get("estimate_if_verified")
    if est:
        lines.append(f"If an inspector verifies a retread, the estimated saving is {est['raw_material_saved_kg']} kg "
                     f"raw material and {est['co2e_avoided_kg']} kg CO2e (Michelin truck/bus factors) [KB:kb-003#2].")
    lines.append("The heatmap shows which regions moved the score; it is an attribution aid, not a defect location.")
    lines.append("\n**Recommended next steps**")
    lines += [f"- {s}" for s in STEP_LABEL[c["triage"]]]
    return "\n".join(lines)


def compose_answer(results: list[dict[str, Any]]) -> str:
    parts = []
    for r in results:
        name, res = r["tool"], r["result"]
        if name == "classify_tyre_image":
            parts.append(compose_classification(res))
        elif name == "get_model_status":
            if res.get("state") != "ready":
                parts.append("No trained model is available yet.")
            else:
                t = {q["id"]: q for q in res["quality_targets"]}
                lt = res["locked_test"]
                parts.append(
                    f"Serving **{res['model_family']} v{res['registry_version']}** (generation {res['generation']}), "
                    f"gate status **{res['gate_status']}**. Development gate cohort: recall "
                    f"{_fmt_pct((t.get('defect_recall') or {}).get('value'))}, specificity "
                    f"{_fmt_pct((t.get('specificity') or {}).get('value'))}. Locked test: recall "
                    f"{_fmt_pct((lt.get('defect_recall') or {}).get('value'))} "
                    f"({(lt.get('defect_recall') or {}).get('numerator')}/{(lt.get('defect_recall') or {}).get('denominator')}), "
                    f"specificity {_fmt_pct((lt.get('specificity') or {}).get('value'))}.")
        elif name == "get_drift_status":
            w = res.get("latest_window")
            if not w:
                parts.append("No monitoring window has been computed yet.")
            else:
                psi = ", ".join(f"{k} {v}" for k, v in w["descriptor_psi"].items())
                parts.append(f"Latest monitoring window: **{w['status']}** over {w['n_inspections']} inspections "
                             f"({w['distinct_parents']} distinct parents). Descriptor PSI: {psi}. Open alerts: "
                             f"{len(res['open_alerts'])}. {res['note'].capitalize()}.")
        elif name == "estimate_circularity_impact":
            parts.append(f"For {res['verified_retreads']} verified retread(s): about {res['raw_material_saved_kg']} kg "
                         f"raw material and {res['co2e_avoided_kg']} kg CO2e avoided (estimate; {res['scope']}) "
                         f"[KB:kb-003#2].")
        elif name == "search_knowledge":
            for p in res["passages"][:2]:
                parts.append(f"{p['text']} [KB:{p['id']}]")
        elif name == "get_review_queue":
            parts.append(f"{res['awaiting_review']} inspection(s) are waiting for a human label.")
        elif name in ("get_inspection", "get_preprocessing_trace") and not res.get("error"):
            if "stages" in res and name == "get_preprocessing_trace":
                parts.append("\n".join(f"{s['index']}. {s['title']} ({s['duration_ms']} ms): " + "; ".join(s["log"][:2])
                                       for s in res["stages"]))
            elif res.get("triage"):
                parts.append(compose_classification(res))
    return "\n\n".join(parts) if parts else (
        "I can classify an attached tyre photo, explain an inspection, report model evidence or drift, and "
        "estimate casing-circularity impact.")


def plan_fallback(text: str, has_attachment: bool) -> list[tuple[str, dict[str, Any]]]:
    low = text.lower()
    calls: list[tuple[str, dict[str, Any]]] = []
    if has_attachment:
        calls.append(("classify_tyre_image", {}))
    m = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", low)
    if m and not has_attachment:
        calls.append(("get_preprocessing_trace" if re.search(r"\b(preprocess\w*|stages?|letterbox\w*|pipeline)\b", low)
                      else "get_inspection", {"inspection_id": m.group(0)}))
    if re.search(r"\b(model|accuracy|recall|specificity|gate|reliab|evidence)", low):
        calls.append(("get_model_status", {}))
    if re.search(r"drift|monitor|alert|psi", low):
        calls.append(("get_drift_status", {}))
    if re.search(r"co2|carbon|sustainab|impact|material|retread", low) and not has_attachment:
        n = re.search(r"\b(\d{1,6})\s*(tyres|tires|casings|retreads)", low)
        calls.append(("estimate_circularity_impact", {"verified_retreads": int(n.group(1)) if n else 1}))
    if re.search(r"queue|waiting|backlog", low):
        calls.append(("get_review_queue", {}))
    if not calls or re.search(r"\b(why|how|what|should|explain|guidance|photo|capture)\b", low):
        calls.append(("search_knowledge", {"query": text[:300]}))
    return calls


def summarize(name: str, res: dict[str, Any]) -> str:
    """One line per tool result for the chat timeline (the full result stays in the trace)."""
    if res.get("error"):
        return f"{name}: {res['error']}" + (f" - {res['message']}" if res.get("message") else "")
    if name == "classify_tyre_image" and res.get("triage"):
        return (f"{res['triage_label']} - calibrated p(defective) {res['p_defective']:.4f}, image quality "
                f"{res['quality_status']}, {res['model']['family']} v{res['model'].get('registry_version')}")
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
    if name == "get_inspection":
        return f"inspection {str(res.get('inspection_id', ''))[:8]}: {res.get('triage_label', res.get('status'))}"
    return "done"


def suggestions_for(results: list[dict[str, Any]], has_attachment: bool) -> list[str]:
    cls = next((r["result"] for r in results if r["tool"] == "classify_tyre_image" and not r["result"].get("error")),
               None)
    if cls:
        base = {"good_candidate": ["What should the retreader still check?", "Estimate the CO2 saving for 25 retreads"],
                "manual_review": ["Which regions drove this score?", "How should I recapture this tyre?"],
                "defect_flagged": ["What visual cues indicate this defect?", "Where does a rejected casing go?"]}
        return [*base[cls["triage"]], f"Show the preprocessing steps for {cls['inspection_id']}",
                "How reliable is the current model?"]
    return ["Upload a tyre photo to classify it", "How reliable is the current model?", "Is there any drift right now?",
            "How much CO2 does retreading save?"]


def build(*, llm, model: str | None, ctx: tools.ToolContext, tracer: Tracer, system_prompt: str,
          history: list[dict[str, Any]], max_iterations: int, llm_params: dict[str, Any]):
    def guard_input(state: AgentState) -> AgentState:
        with tracer.span("guard_input", "guard"):
            v = guardrails.check_input(state["user_text"], state["has_attachment"])
            tracer.guard_events += v.events
        flagged = [e.rule for e in v.events if e.action != "pass"]
        ctx.emit("guard_input", {"status": "blocked" if v.blocked else "passed", "rules": flagged,
                                 "message": "input guardrails: " + ("blocked (" + ", ".join(flagged) + ")"
                                                                    if v.blocked else
                                                                    "passed" + (f" ({', '.join(flagged)})" if flagged else ""))})
        if v.blocked:
            return {"blocked": True, "final_text": v.reply or "Request blocked."}
        user_content = v.text or "Please classify the attached tyre photo."
        if state["has_attachment"]:
            user_content += "\n\n[A tyre photo is attached to this message.]"
        msgs = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_content}]
        if v.certification_request:
            msgs.insert(1, {"role": "system", "content": "The user is asking for a safety or certification "
                            "judgement. Explain that the model cannot certify tyres and route to an inspector."})
        return {"blocked": False, "messages": msgs, "certification_request": v.certification_request,
                "iterations": 0, "mode": "llm" if llm and model else "deterministic_fallback"}

    def plan(state: AgentState) -> AgentState:
        it = state.get("iterations", 0)
        ctx.emit("plan", {"mode": "llm", "model": model, "iteration": it + 1,
                          "message": f"{model} is deciding the next step" + (" (photo attached: YOLO tool first)"
                                                                          if state["has_attachment"] and it == 0 else "")})
        with tracer.span("llm_plan", "llm", iteration=it, model=model) as attrs:
            try:
                force_tool = state["has_attachment"] and it == 0
                msg = llm.chat_tools(model, state["messages"], tools.SCHEMAS,
                                     tool_choice={"type": "function", "function": {"name": "classify_tyre_image"}}
                                     if force_tool else "auto",
                                     temperature=llm_params["temperature"], max_tokens=llm_params["max_tokens"],
                                     timeout=llm_params["timeout_s"])
                tracer.add_usage(llm.last_usage, model)
                attrs["usage"] = llm.last_usage
            except Exception as exc:
                attrs["error"] = f"{exc.__class__.__name__}: {str(exc)[:200]}"
                ctx.emit("plan", {"mode": "llm", "error": attrs["error"],
                                  "message": "LLM call failed; switching to the deterministic tool-grounded planner"})
                return {"mode": "deterministic_fallback", "llm_error": attrs["error"]}
        calls = msg.get("tool_calls") or []
        clean: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
        if calls:
            clean["tool_calls"] = [{"id": c["id"], "type": "function",
                                    "function": {"name": c["function"]["name"],
                                                 "arguments": c["function"]["arguments"] or "{}"}} for c in calls]
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
                attrs["result_keys"] = sorted(res)[:20]
                attrs["error"] = res.get("error")
                ctx.emit("tool_result", {"tool": name, "ok": not res.get("error"), "summary": summarize(name, res)})
            msgs.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(res)[:12000]})
        return {"messages": msgs, "pending_calls": []}

    def fallback(state: AgentState) -> AgentState:
        reason = state.get("llm_error") or "no LLM configured"
        with tracer.span("deterministic_planner", "node", reason=reason):
            done = {r["tool"] for r in ctx.results}
            planned = [(n, a) for n, a in plan_fallback(state["user_text"], state["has_attachment"]) if n not in done]
            ctx.emit("plan", {"mode": "deterministic_fallback", "tools": [n for n, _ in planned],
                              "message": f"tool-grounded planner ({reason}) chose: " + ", ".join(n for n, _ in planned)})
            for name, args in planned:
                with tracer.span(f"tool:{name}", "tool", arguments=args):
                    ctx.emit("tool_call", {"tool": name, "args": json.dumps(args)})
                    res = tools.execute(ctx, name, args)
                    tracer.tools_used.append(name)
                    ctx.emit("tool_result", {"tool": name, "ok": not res.get("error"), "summary": summarize(name, res)})
            ctx.emit("compose", {"message": "composing the answer from tool results"})
            text = compose_answer(ctx.results)
        return {"final_text": text, "mode": "deterministic_fallback"}

    def guard_output(state: AgentState) -> AgentState:
        cls = next((r["result"] for r in ctx.results if r["tool"] == "classify_tyre_image"
                    and not r["result"].get("error")), None)
        with tracer.span("guard_output", "guard"):
            v = guardrails.check_output(state.get("final_text") or "", tool_results=[r["result"] for r in ctx.results],
                                        system_prompt=system_prompt, classification=cls,
                                        fallback_summary=compose_answer(ctx.results) if ctx.results else None)
            tracer.guard_events += v.events
        flagged = [e.rule for e in v.events if e.action != "pass"]
        g = v.grounding
        ctx.emit("guard_output", {"rules": flagged, "grounding": g,
                                  "message": f"output guardrails: {g.get('checked', 0) - len(g.get('ungrounded', []))}/"
                                             f"{g.get('checked', 0)} figures grounded in tool outputs"
                                             + (f"; {', '.join(flagged)}" if flagged else "")})
        return {"final_text": v.text, "grounding": v.grounding}

    def suggest(state: AgentState) -> AgentState:
        return {"suggestions": suggestions_for(ctx.results, state["has_attachment"])}

    def after_guard(state: AgentState) -> str:
        if state.get("blocked"):
            return "suggest"
        return "plan" if state.get("mode") == "llm" else "fallback"

    def after_plan(state: AgentState) -> str:
        if state.get("mode") != "llm":
            return "fallback"
        if state.get("pending_calls") and state.get("iterations", 0) <= max_iterations:
            return "tools"
        if not (state.get("final_text") or "").strip():
            return "fallback"
        return "guard_output"

    g = StateGraph(AgentState)
    g.add_node("guard_input", guard_input)
    g.add_node("plan", plan)
    g.add_node("tools", run_tools)
    g.add_node("fallback", fallback)
    g.add_node("guard_output", guard_output)
    g.add_node("suggest", suggest)
    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", after_guard, {"plan": "plan", "fallback": "fallback", "suggest": "suggest"})
    g.add_conditional_edges("plan", after_plan, {"tools": "tools", "fallback": "fallback",
                                                 "guard_output": "guard_output"})
    g.add_edge("tools", "plan")
    g.add_edge("fallback", "guard_output")
    g.add_edge("guard_output", "suggest")
    g.add_edge("suggest", END)
    return g.compile()
