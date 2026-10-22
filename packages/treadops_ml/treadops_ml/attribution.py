"""Occlusion-sensitivity attribution and the annotated "final classified image".

Model-agnostic: a grid of patches is replaced by the letterbox pad value and the calibrated p_defective is
re-measured in one batch. Positive cells are regions whose removal lowers p_defective (evidence for
"defective"). This is an attribution heatmap - not a defect segmentation and not a detection box.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TRIAGE_COLOURS = {
    "good_candidate": (16, 150, 120),
    "manual_review": (217, 140, 20),
    "defect_flagged": (214, 51, 72),
}
TRIAGE_TEXT = {
    "good_candidate": "RETREAD CANDIDATE",
    "manual_review": "EXPERT REVIEW",
    "defect_flagged": "DEFECT FLAGGED",
}


def occlusion_map(bundle, x: np.ndarray, grid: int = 8) -> dict[str, Any]:
    s = x.shape[-1]
    cell = s // grid
    fill = bundle.preprocess.pad_value * bundle.preprocess.scale
    batch = np.repeat(x, grid * grid, axis=0)
    for i in range(grid):
        for j in range(grid):
            batch[i * grid + j, :, i * cell : (i + 1) * cell, j * cell : (j + 1) * cell] = fill
    base = float(bundle.p_defective(bundle.raw_probs(x))[0])
    occluded = bundle.p_defective(bundle.raw_probs(batch)).reshape(grid, grid)
    return {"base_p_defective": base, "delta": (base - occluded).astype(np.float32), "grid": grid, "cell": cell}


def render_heatmap(canonical_rgb: np.ndarray, delta: np.ndarray, lmeta: dict, alpha: float = 0.45) -> Image.Image:
    """Project the letterbox-space grid back onto the canonical image and blend a diverging colour map."""
    s = lmeta["size"]
    full = cv2.resize(delta, (s, s), interpolation=cv2.INTER_CUBIC)
    crop = full[lmeta["top"] : lmeta["top"] + lmeta["resized_h"], lmeta["left"] : lmeta["left"] + lmeta["resized_w"]]
    h, w = canonical_rgb.shape[:2]
    m = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
    lim = max(float(np.abs(delta).max()), 1e-3)
    v = np.clip(m / lim, -1, 1)
    red = np.array([214, 51, 72], np.float32)
    blue = np.array([45, 110, 200], np.float32)
    colour = np.where(v[..., None] >= 0, red, blue)
    weight = (np.abs(v)[..., None] * alpha).astype(np.float32)
    out = canonical_rgb.astype(np.float32) * (1 - weight) + colour * weight
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def annotate(img: Image.Image, triage: str, p_def: float, model_label: str, side: int = 720) -> Image.Image:
    """Final classified image: border + banner in the triage colour. No invented boxes."""
    img = img.copy()
    img.thumbnail((side, side), Image.Resampling.LANCZOS)
    colour = TRIAGE_COLOURS[triage]
    border, banner = 10, 54
    canvas = Image.new("RGB", (img.width + 2 * border, img.height + 2 * border + banner), colour)
    canvas.paste(img, (border, border + banner))
    d = ImageDraw.Draw(canvas)
    d.text((border + 6, 8), TRIAGE_TEXT[triage], fill=(255, 255, 255), font=_font(24))
    d.text((border + 6, 34), f"p(defective)={p_def:.3f} · {model_label}", fill=(255, 255, 255), font=_font(14))
    return canvas
