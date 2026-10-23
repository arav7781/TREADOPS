"""Vision second opinion: a multimodal LLM re-checks every tyre the YOLO policy would auto-clear.

The YOLO classifier learned mostly "weathered vs new rubber" from its training data, so a fresh tyre with a single
cut can score low p(defective). Before such a tyre is shown as "not damaged", a vision model looks at the same
canonical image. If it sees damage (medium or high confidence) or gives no usable answer, the tyre goes to an expert
check instead; it never downgrades a YOLO "damaged". Measured on the locked test split (277 images): defective
tyres auto-cleared fell from 3/150 to 1/150 while 26/127 good tyres went to an expert (docs/evidence/vision_check).

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
    return {"damage_visible": data["damage_visible"],
            "confidence": str(data.get("confidence", "")).lower() or None,
            "findings": [str(f)[:80] for f in (data.get("findings") or [])][:3],
            "location": data.get("location")}


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
    return {**meta, **out, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
