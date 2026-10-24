"""Vision second opinion: a multimodal LLM re-checks every tyre the YOLO policy would auto-clear (and every image
the subject gate finds unusual), answering two questions: is a tyre the subject at all, and is damage visible.

The YOLO classifier learned mostly "weathered vs new rubber" from its training data, so a fresh tyre with a single
cut can score low p(defective). Before such a tyre is shown as "not damaged", a vision model looks at the same
canonical image. If it sees damage (medium or high confidence) or gives no usable answer, the tyre goes to an expert
check instead; it never downgrades a YOLO "damaged". Measured on the locked test split (277 images, prompt v2):
defective tyres auto-cleared fell from 3/150 to 1/150 while 28 of 116 auto-cleared good tyres went to an expert
(docs/evidence/vision_check). Its "no tyre" answer rejects an image only together with the subject gate.

Enabled by VISION_CHECK_MODEL (an OpenAI-compatible vision model on LLM_BASE_URL); images leave the network.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

from treadops_api.settings import get_settings
from treadops_ml.common import load_yaml, sha256_bytes

ESCALATE_CONFIDENCE = {"medium", "high"}
TIMEOUT_S = 25
BREAKER_S = 300.0  # after the provider is unreachable or times out, skip it (fail-safe: expert check) this long
_breaker: dict[str, Any] = {"open_until": 0.0, "reason": ""}


def _reachable(base_url: str | None, timeout: float = 1.5) -> bool:
    """A TCP connect to the provider: a dead gateway costs 1.5 s instead of two 25 s request timeouts."""
    import socket
    from urllib.parse import urlsplit

    u = urlsplit(base_url or "")
    if not u.hostname:
        return True  # let the client report the misconfiguration
    try:
        socket.create_connection((u.hostname, u.port or (443 if u.scheme == "https" else 80)), timeout=timeout).close()
        return True
    except OSError:
        return False


@lru_cache
def prompt() -> tuple[str, str, str]:
    p = load_yaml("configs/prompts/vision_check.yaml")
    return p["template"], p["version"], sha256_bytes(p["template"].encode())


def enabled() -> bool:
    s = get_settings()
    return bool(s.vision_check_model and s.llm_base_url)


def parse(text: str) -> dict[str, Any]:
    """The model's JSON answer, or an error marker when there is none (treated as 'no usable answer')."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("damage_visible"), bool):
        return {"error": "unparseable_answer", "raw": (text or "")[:200]}
    tyre = data.get("tyre_visible")
    return {"tyre_visible": tyre if isinstance(tyre, bool) else None,
            "subject": str(data.get("subject") or "")[:80] or None,
            "damage_visible": data["damage_visible"],
            "confidence": str(data.get("confidence", "")).lower() or None,
            "findings": [str(f)[:80] for f in (data.get("findings") or [])][:3],
            "location": data.get("location")}


def sees_no_tyre(opinion: dict[str, Any] | None) -> bool:
    """True only for an explicit "no tyre in this photo"; a missing or failed answer never rejects an image."""
    return bool(opinion) and opinion.get("tyre_visible") is False  # type: ignore[union-attr]


def decide(policy_triage: str, opinion: dict[str, Any] | None) -> tuple[str, str, str | None]:
    """(final triage, decision, reason). Only an auto-clear can change, and only towards an expert check."""
    if policy_triage != "good_candidate" or opinion is None:
        return policy_triage, "not_needed", None
    if "damage_visible" not in opinion:
        return "manual_review", "unavailable", "vision second opinion unavailable, so not auto-cleared"
    if opinion["damage_visible"] and opinion.get("confidence") in ESCALATE_CONFIDENCE:
        seen = ", ".join(opinion.get("findings") or []) or "damage"
        return "manual_review", "disagrees", f"vision second opinion sees damage ({seen}); models disagree"
    return "good_candidate", "agrees", None


def second_opinion(canonical_rgb: np.ndarray) -> dict[str, Any]:
    from openai import OpenAI

    s = get_settings()
    template, version, sha = prompt()
    img = Image.fromarray(canonical_rgb)
    img.thumbnail((768, 768))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    meta = {"model": s.vision_check_model, "prompt_version": version, "prompt_sha256": sha}
    if time.monotonic() < _breaker["open_until"]:  # a line camera must not wait 50 s per frame on a dead provider
        return {**meta, "error": f"skipped: provider unreachable moments ago ({_breaker['reason']})", "latency_ms": 0.0}
    if not _reachable(s.llm_base_url):
        _breaker.update(open_until=time.monotonic() + BREAKER_S, reason="no connection")
        return {**meta, "error": "provider unreachable (no connection)", "latency_ms": 1500.0}
    t0 = time.perf_counter()
    try:
        client = OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key or "not-needed", max_retries=1)
        r = client.chat.completions.create(
            model=s.vision_check_model or "", temperature=0, max_tokens=300, timeout=TIMEOUT_S,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": template},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}])
        out = parse(r.choices[0].message.content or "")
    except Exception as exc:  # any provider failure means "no usable answer", never an auto-clear
        out = {"error": f"{exc.__class__.__name__}: {str(exc)[:160]}"}
        if exc.__class__.__name__ in ("APITimeoutError", "APIConnectionError"):
            _breaker.update(open_until=time.monotonic() + BREAKER_S, reason=exc.__class__.__name__)
    return {**meta, **out, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
