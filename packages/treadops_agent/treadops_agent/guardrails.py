"""Copilot guardrails. Pure functions: every decision is returned as a list of typed events.

Input:  length, prompt injection (block), PII redaction (email/phone/vehicle reg/card), scope (block off-topic)
Output: certification claims (rewrite), ungrounded numbers (flag + note), a verdict on a rejected non-tyre image
        (replace), triage contradicting the YOLO tool
        (replace with the grounded summary), system-prompt leakage (block), PII (redact), mandatory disclaimer
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from treadops_ml.common import load_yaml

PII_PATTERNS = {  # card before phone: a phone match would otherwise take a card's first digits
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "card_number": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,11}\d(?!\d)"),
    "vehicle_registration": re.compile(r"\b[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,3}[ -]?\d{4}\b"),
}
# Inspection IDs and hashes are not PII: the digits between a UUID's hyphens look like a phone number.
IDENTIFIER = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
                        r"|\b(?=[0-9a-f]*[a-f])[0-9a-f]{12,64}\b", re.IGNORECASE)
NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d+)?)\s?%|(?<![\w.])(0\.\d{2,})(?![\d])")
TRIAGE_WORDS = {
    "good_candidate": ("retread candidate", "good candidate", "good_candidate"),
    "manual_review": ("expert review", "manual review", "manual_review"),
    "defect_flagged": ("defect flagged", "defect_flagged", "flagged as defective"),
}
# Damage verdict wording. "not damaged" is matched first so it never also counts as "damaged".
VERDICT_NOT = re.compile(r"\b(not damaged|undamaged|no (?:visible )?damage|isn't damaged)\b")
VERDICT_YES = re.compile(r"\bdamaged\b")
# Talking about the verdict itself ("no damage verdict was produced") is not claiming one.
ABOUT_THE_VERDICT = re.compile(r"\b(?:no |any |a )?damage (?:verdict|assessment|check|classification|judge?ment|result)s?\b")
UNCERTAIN_WORDS = re.compile(r"\b(uncertain|expert (?:check|review)|manual review|disagree\w*)\b")


def claimed_verdict(low: str) -> str | None:
    says_not = bool(VERDICT_NOT.search(low))
    says_yes = bool(VERDICT_YES.search(VERDICT_NOT.sub(" ", low)))
    if says_not != says_yes:
        return "not_damaged" if says_not else "damaged"
    return None


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
        keep = [m.span() for m in IDENTIFIER.finditer(text)]
        hits = [m for m in pat.finditer(text) if not any(m.start() < e and s < m.end() for s, e in keep)]
        for m in reversed(hits):  # right to left so earlier offsets stay valid
            text = text[: m.start()] + f"[{name.upper()}]" + text[m.end():]
        if hits:
            counts[name] = len(hits)
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
        if isinstance(o, int | float):  # figures in the answer are read without their sign, so compare magnitudes
            vals.append(abs(float(o)))
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


CLAUSE_END = re.compile(r"[.!?;:](?=\s)|\n|\b(?:but|however|and|so)\b")
SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")
LINE_MARKER = re.compile(r"\s*(?:[-*+>#]+\s+|\d+[.)]\s+)?")
# Negated ("does not certify") or stated as a requirement ("must be certified by an inspector"): not a claim.
NOT_A_CLAIM = re.compile(r"\b(not|never|cannot|can't|isn't|no|without|only|requires?|required|needs?|must|until|"
                         r"unless|pending|before)\b")
CLEARANCE_SENTENCE = "Only a qualified inspector can clear this tyre for use."


def _clause_before(low: str, pos: int) -> str:
    """The text before `pos` back to the start of its clause (sentence end, ;, :, line break, but/and/so)."""
    ends = [m.end() for m in CLAUSE_END.finditer(low, 0, pos)]
    return low[ends[-1] if ends else 0 : pos]


def _is_claim(low: str, m: re.Match[str], qualifiers: list[str]) -> bool:
    """A certification phrase is a claim unless negated or a requirement, or it describes who does the work
    ("certified inspector", "certified recycling")."""
    if NOT_A_CLAIM.search(_clause_before(low, m.start())):
        return False
    return not re.match(rf"\s+(?:tyre\s+|tire\s+)?(?:{'|'.join(qualifiers)})", low[m.end():])


def _sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
    """The sentence around text[start:end], keeping any list, quote or heading marker in front of it."""
    before = [m.end() for m in SENTENCE_END.finditer(text, 0, start)]
    s = before[-1] if before else 0
    s = LINE_MARKER.match(text, s).end()  # type: ignore[union-attr]  # the pattern can match empty
    after = SENTENCE_END.search(text, end)
    e = len(text) if after is None else after.start() if after.group() == "\n" else after.end()
    return s, e


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

    # 2) triage contradiction with the YOLO tool; no verdict at all for an image the subject gate rejected
    replaced = False
    if classification and classification.get("verdict") == "not_a_tyre":
        plain = ABOUT_THE_VERDICT.sub(" ", low)
        claimed = {t for t, words in TRIAGE_WORDS.items() if any(w in plain for w in words)}
        said = claimed_verdict(plain)
        if claimed or said:
            events.append(GuardEvent("output", "verdict_on_non_tyre", "rewrite",
                                     {"claimed": sorted(claimed), "claimed_verdict": said}))
            if fallback_summary:
                text, low, replaced = fallback_summary, fallback_summary.lower(), True
    elif classification and classification.get("triage"):
        actual = classification["triage"]
        claimed = {t for t, words in TRIAGE_WORDS.items() if any(w in low for w in words)}
        verdict, said = classification.get("verdict"), claimed_verdict(low)
        # An uncertain result may be described as possibly damaged, or quote YOLO's own "not damaged" alongside the
        # expert check, but never presented as simply not damaged.
        wrong_verdict = (said == "not_damaged" and not UNCERTAIN_WORDS.search(low)) if verdict == "uncertain" \
            else bool(verdict and said and said != verdict)
        if (claimed and actual not in claimed) or wrong_verdict:
            events.append(GuardEvent("output", "triage_contradiction", "rewrite",
                                     {"tool_triage": actual, "claimed": sorted(claimed), "tool_verdict": verdict,
                                      "claimed_verdict": said}))
            if fallback_summary:
                text, low, replaced = fallback_summary, fallback_summary.lower(), True

    # 3) certification / safety claims: the whole sentence making one is replaced, so the answer still reads well.
    spans: list[tuple[int, int]] = []
    for pat in g["certification_patterns"]:
        claims = [m for m in re.finditer(pat, text, flags=re.IGNORECASE)
                  if _is_claim(low, m, g.get("certification_qualifiers", []))]
        if claims:
            events.append(GuardEvent("output", "certification_claim", "rewrite", {"pattern": pat, "count": len(claims)}))
            spans += [_sentence_span(text, m.start(), m.end()) for m in claims]
    merged: list[tuple[int, int]] = []
    for s, e in sorted(set(spans)):
        if merged and s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    for s, e in reversed(merged):  # right to left so earlier offsets stay valid
        text = text[:s] + CLEARANCE_SENTENCE + text[e:]
    low = text.lower()

    # 3b) certification vocabulary, even when negated ("does not mean it is roadworthy" -> "fit for use")
    reworded = 0
    for pat, plain in g.get("certification_vocabulary", {}).items():
        text, n = re.subn(rf"\b{pat}\b", plain, text, flags=re.IGNORECASE)
        reworded += n
    if reworded:
        events.append(GuardEvent("output", "certification_vocabulary", "rewrite", {"count": reworded}))
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
