"""Copilot guardrails. Pure functions: every decision is returned as a list of typed events.

Input:  length, prompt injection (block), PII redaction (email/phone/vehicle reg/card), scope (block off-topic)
Output: certification claims (rewrite), ungrounded numbers (flag + note), triage contradicting the YOLO tool
        (replace with the grounded summary), system-prompt leakage (block), PII (redact), mandatory disclaimer
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from treadops_ml.common import load_yaml

PII_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,11}\d(?!\d)"),
    "vehicle_registration": re.compile(r"\b[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,3}[ -]?\d{4}\b"),
    "card_number": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}
NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d+)?)\s?%|(?<![\w.])(0\.\d{2,})(?![\d])")
TRIAGE_WORDS = {
    "good_candidate": ("retread candidate", "good candidate", "good_candidate"),
    "manual_review": ("expert review", "manual review", "manual_review"),
    "defect_flagged": ("defect flagged", "defect_flagged", "flagged as defective"),
}


@lru_cache
def config() -> dict[str, Any]:
    return load_yaml("configs/copilot.yaml")


@dataclass
class GuardEvent:
    stage: str
    rule: str
    action: str  # block | redact | rewrite | warn | pass
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class InputVerdict:
    text: str
    blocked: bool
    events: list[GuardEvent]
    reply: str | None = None
    certification_request: bool = False


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for name, pat in PII_PATTERNS.items():
        text, n = pat.subn(f"[{name.upper()}]", text)
        if n:
            counts[name] = n
    return text, counts


def check_input(text: str, has_attachment: bool) -> InputVerdict:
    cfg = config()
    g = cfg["guardrails"]
    events: list[GuardEvent] = []
    if len(text) > cfg["max_message_chars"]:
        events.append(GuardEvent("input", "max_length", "block", {"chars": len(text)}))
        return InputVerdict(text, True, events, f"Messages are limited to {cfg['max_message_chars']} characters.")
    low = text.lower()
    for pat in g["injection_patterns"]:
        if re.search(pat, low):
            events.append(GuardEvent("input", "prompt_injection", "block", {"pattern": pat}))
            return InputVerdict(text, True, events,
                                "I can't change my instructions or reveal them. I can help with tyre inspection "
                                "results, model evidence, monitoring and casing circularity.")
    clean, pii = redact(text)
    if pii:
        events.append(GuardEvent("input", "pii_redaction", "redact", {"counts": pii}))
    cert = any(re.search(p, low) for p in g["certification_patterns"])
    if cert:
        events.append(GuardEvent("input", "certification_request", "warn", {}))
    words = re.findall(r"[a-z0-9]+", low)
    on_topic = has_attachment or not words or any(any(w.startswith(k) for k in g["domain_keywords"]) for w in words)
    if not on_topic and len(words) > 3:
        events.append(GuardEvent("input", "off_topic", "block", {"words": len(words)}))
        return InputVerdict(clean, True, events,
                            "That's outside what I can help with here. Ask me about a tyre photo, an inspection, "
                            "the model's evidence, drift monitoring, or casing circularity.")
    if not events:
        events.append(GuardEvent("input", "all_checks", "pass", {}))
    return InputVerdict(clean, False, events, certification_request=cert)


def _numbers(text: str) -> list[float]:
    out = []
    for pct, dec in NUMBER.findall(text):
        if pct:
            out.append(float(pct) / 100.0)
        elif dec:
            out.append(float(dec))
    return out


def allowed_numbers(tool_results: list[dict[str, Any]]) -> list[float]:
    vals: list[float] = []

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, int | float):
            vals.append(float(o))
        elif isinstance(o, str):  # figures quoted from stage logs ("r=0.2188", "scale 0.341") are tool output too
            vals.extend(float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", o))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(tool_results)
    return vals


def grounded(value: float, allowed: list[float]) -> bool:
    """A figure is grounded if it matches some tool value as a fraction or percentage, after rounding."""
    for a in allowed:
        for cand in (a, a / 100.0):
            if abs(cand - value) <= 0.006 or (cand and abs(cand - value) / abs(cand) < 0.01):
                return True
    return False


def _clause_before(low: str, pos: int, width: int = 40) -> str:
    """Up to `width` characters before `pos`, never reaching back past the start of the sentence."""
    window = low[max(0, pos - width) : pos]
    cut = max(window.rfind(c) for c in ".!?\n")
    return window[cut + 1 :] if cut >= 0 else window


@dataclass
class OutputVerdict:
    text: str
    events: list[GuardEvent]
    grounding: dict[str, Any]
    replaced: bool = False


def check_output(text: str, *, tool_results: list[dict[str, Any]], system_prompt: str,
                 classification: dict[str, Any] | None, fallback_summary: str | None) -> OutputVerdict:
    cfg = config()
    g = cfg["guardrails"]
    events: list[GuardEvent] = []
    low = text.lower()

    # 1) leakage of the system prompt (any 60-char window verbatim)
    sp = " ".join(system_prompt.split())
    flat = " ".join(text.split())
    for i in range(0, max(1, len(sp) - 60), 30):
        if sp[i : i + 60] and sp[i : i + 60] in flat:
            events.append(GuardEvent("output", "system_prompt_leak", "block", {}))
            return OutputVerdict("I can't share my configuration, but I'm happy to help with the inspection.",
                                 events, {"checked": 0, "ungrounded": []}, replaced=True)

    # 2) triage contradiction with the YOLO tool
    replaced = False
    if classification and classification.get("triage"):
        actual = classification["triage"]
        claimed = {t for t, words in TRIAGE_WORDS.items() if any(w in low for w in words)}
        if claimed and actual not in claimed:
            events.append(GuardEvent("output", "triage_contradiction", "rewrite",
                                     {"tool_triage": actual, "claimed": sorted(claimed)}))
            if fallback_summary:
                text, low, replaced = fallback_summary, fallback_summary.lower(), True

    # 3) certification / safety claims
    for pat in g["certification_patterns"]:
        # Rewrite only the un-negated matches, right to left so earlier offsets stay valid.
        claims = [m for m in re.finditer(pat, text, flags=re.IGNORECASE)
                  if not re.search(r"\b(not|never|cannot|can't|isn't|no)\b", _clause_before(low, m.start()))]
        if claims:
            events.append(GuardEvent("output", "certification_claim", "rewrite", {"pattern": pat, "count": len(claims)}))
            for m in reversed(claims):
                text = text[: m.start()] + "[requires inspector clearance]" + text[m.end():]
            low = text.lower()

    # 4) grounding of figures
    allowed = allowed_numbers(tool_results)
    nums = _numbers(text)
    ungrounded = [n for n in nums if not grounded(n, allowed)]
    if ungrounded:
        events.append(GuardEvent("output", "ungrounded_number", "warn", {"values": ungrounded[:10]}))
        text += "\n\n_Note: some figures above could not be matched to tool outputs; verify them in the evidence views._"

    # 5) PII in output
    text, pii = redact(text)
    if pii:
        events.append(GuardEvent("output", "pii_redaction", "redact", {"counts": pii}))

    # 6) mandatory disclaimer whenever a classification is discussed
    if classification and g["disclaimer"].lower()[:30] not in text.lower():
        text = f"{text.rstrip()}\n\n> {g['disclaimer']}"
        events.append(GuardEvent("output", "disclaimer", "rewrite", {}))
    if not events:
        events.append(GuardEvent("output", "all_checks", "pass", {}))
    return OutputVerdict(text, events, {"checked": len(nums), "ungrounded": ungrounded,
                                        "pass": not ungrounded}, replaced=replaced)
