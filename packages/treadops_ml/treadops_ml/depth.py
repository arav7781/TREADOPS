"""Advisory depth stage: monocular relative depth of the canonical photo (Depth Anything V2 Small, ONNX).

An encoder (a ViT) turns the photo into patch features; a DPT decoder fuses them back into a dense map of
*relative* inverse depth: larger means nearer, with no unit and no scale. It is not metric depth, so it cannot
measure tread depth in millimetres, and at this resolution it smooths away cuts and cracks (checked on a sidewall
gash and a bulge). What it shows reliably is the scene's geometry: which surface is nearest the camera (normally
the tyre) and how much of the frame it fills, a framing check that matters for wide line-camera shots.

The stage is shown to the inspector and recorded; it never changes the verdict. Settings: configs/depth.yaml.
"""

from __future__ import annotations

import threading
import time
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from .common import load_yaml
from .weights import pinned_sha, pretrained

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
PATCH = 14  # the ViT encoder's patch size: both input sides must be multiples of it


@lru_cache
def config() -> dict[str, Any]:
    return load_yaml("configs/depth.yaml")


class DepthEstimator:
    def __init__(self, weights: str | None = None, input_px: int | None = None):
        import onnxruntime as ort

        cfg = config()
        self.weights = weights or cfg["model"]
        self.input_px = int(input_px or cfg["input_px"])
        self.session = ort.InferenceSession(str(pretrained(self.weights)), providers=["CPUExecutionProvider"])
        self._lock = threading.Lock()

    def identity(self) -> dict[str, Any]:
        return {"model": config()["display_name"], "weights": self.weights, "sha256": pinned_sha(self.weights),
                "input_px": self.input_px, "license": config()["license"]}

    def relative_depth(self, rgb: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 RGB -> HxW float32 relative inverse depth (larger = nearer), at the input's size."""
        h, w = rgb.shape[:2]
        k = self.input_px / max(h, w)
        nh, nw = max(PATCH, round(h * k / PATCH) * PATCH), max(PATCH, round(w * k / PATCH) * PATCH)
        x = (cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 - MEAN) / STD
        with self._lock:
            out = self.session.run(None, {"pixel_values": x.transpose(2, 0, 1)[None]})[0][0]
        return cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)

    def stage(self, rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """The rendered stage image (depth colour map with the near-surface outline) and its recorded facts."""
        t0 = time.perf_counter()
        d = self.relative_depth(rgb)
        model_ms = (time.perf_counter() - t0) * 1000
        n = (d - d.min()) / max(float(np.ptp(d)), 1e-6)
        u8 = (n * 255).astype(np.uint8)
        _, near = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # nearest surface vs the rest
        near = cv2.morphologyEx(near, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        fraction = float((near > 0).mean())
        img = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
        contours, _ = cv2.findContours(near, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 229, 252), max(2, max(rgb.shape[:2]) // 300))  # brand yellow (BGR)
        box = None  # the largest near region, normalised [x0, y0, x1, y1]: where the tyre is in the frame
        if contours:
            x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
            h, w = rgb.shape[:2]
            box = [round(x / w, 4), round(y / h, 4), round((x + bw) / w, 4), round((y + bh) / h, 4)]
        framing = config()["framing"]
        verdict = ("tyre fills the frame" if fraction >= framing["fills_frame"] else
                   "tyre is small in the frame: move closer" if fraction < framing["too_small"] else
                   "tyre partly fills the frame")
        facts = {**self.identity(), "relative": True, "units": None, "near_surface_fraction": round(fraction, 4),
                 "near_surface_box": box,
                 "framing": verdict, "depth_ms": round(model_ms, 1), "used_for_verdict": False,
                 "limits": config()["limits"]}
        return img[:, :, ::-1], facts
