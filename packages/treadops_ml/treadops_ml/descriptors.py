"""Stable numeric image descriptors, perceptual hashing and a fixed reference embedding encoder.

Descriptors are computed on the canonical RGB image after resizing the long side to 512 px so they are
comparable across source resolutions. The reference encoder is the pretrained ImageNet checkpoint pinned by
SHA-256 in the bundle manifest - never the served (fine-tuned) model - so the drift feature space does not
move when a challenger is promoted.
"""

from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np
from scipy.fft import dctn

DESCRIPTOR_VERSION = "v1"
DESCRIPTOR_NAMES = ("brightness", "contrast", "blur_variance", "saturation", "aspect_ratio")


def _norm_gray(rgb: np.ndarray, side: int = 512) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    r = side / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * r)), max(1, round(h * r))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return small, gray


def image_descriptors(rgb: np.ndarray) -> dict[str, float]:
    small, gray = _norm_gray(rgb)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    return {
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "blur_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "saturation": float(hsv[..., 1].mean() / 255.0),
        "aspect_ratio": float(rgb.shape[1] / rgb.shape[0]),
        "width": float(rgb.shape[1]),
        "height": float(rgb.shape[0]),
    }


def quality_status(desc: dict[str, float], policy_quality: dict[str, float]) -> tuple[str, list[str]]:
    reasons = []
    if desc["brightness"] < policy_quality["min_brightness"]:
        reasons.append("underexposed")
    if desc["brightness"] > policy_quality["max_brightness"]:
        reasons.append("overexposed")
    if desc["blur_variance"] < policy_quality["min_blur_variance"]:
        reasons.append("blurred")
    return ("accepted" if not reasons else "low_quality"), reasons


def phash(rgb: np.ndarray) -> int:
    """64-bit DCT perceptual hash (32x32 grayscale, top-left 8x8 low frequencies, median threshold)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float64)
    low = dctn(small, norm="ortho")[:8, :8]
    bits = (low > np.median(low[1:].ravel() if low.size > 1 else low)).ravel()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class ReferenceEncoder:
    """Pooled penultimate features of the pinned pretrained classifier, on the shared letterbox tensor."""

    _lock = threading.Lock()

    def __init__(self, weights: str = "yolo26n-cls.pt", device: str = "cpu"):
        from ultralytics import YOLO

        self.weights = weights
        self.device = device
        self.model = YOLO(weights, task="classify")
        self.net = self.model.model.to(device).eval()
        self._feat: Any = None
        head = self.net.model[-1]  # Classify head: conv -> pool -> drop -> linear
        head.pool.register_forward_hook(lambda _m, _i, out: setattr(self, "_feat", out))

    @property
    def dim(self) -> int:
        return int(self.net.model[-1].linear.in_features)

    def embed(self, x: np.ndarray) -> np.ndarray:
        import torch

        with self._lock, torch.inference_mode():
            self.net(torch.from_numpy(x).to(self.device))
            feat = self._feat.flatten(1).float().cpu().numpy()
        norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12
        return (feat / norms).astype(np.float32)

    def identity(self) -> dict[str, Any]:
        from .common import sha256_file

        ckpt = self.model.ckpt_path or self.weights
        return {"weights": self.weights, "sha256": sha256_file(ckpt), "dim": self.dim, "pooling": "head.pool+L2"}
