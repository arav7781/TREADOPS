"""Deterministic image corruptions shared by stress evaluation and the operational replay simulator.

A corrupted image is a *derivative*: it keeps its parent image ID, label and role, and its transform hash.
The output is re-encoded JPEG so it enters the full serving pipeline exactly like an upload would.
"""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .common import sha256_json


def apply_transform(rgb: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    out = rgb
    if "brightness_factor" in transform:
        out = np.clip(out.astype(np.float32) * float(transform["brightness_factor"]), 0, 255).astype(np.uint8)
    if "gaussian_sigma" in transform:
        s = float(transform["gaussian_sigma"])
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=s, sigmaY=s)
    return out


def derivative_bytes(canonical_rgb: np.ndarray, transform: dict[str, Any], quality: int = 95) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(apply_transform(canonical_rgb, transform)).save(buf, format="JPEG", quality=quality,
                                                                     subsampling=0)
    return buf.getvalue()


def transform_hash(transform: dict[str, Any]) -> str:
    return sha256_json({"transform": transform, "encoder": "JPEG q95 4:4:4"})
