"""Subject gate: is this image like the tyre photos the model was trained on?

The fixed drift encoder (pinned yolo26n-cls) embeds every image. The score is the distance of that embedding from the
bundle's tyre reference subspace (PCA fitted on the calibration and policy photos): the residual after projection.
Above `unusual_above` the image is unlike the training tyres, so the YOLO verdict is not trusted: it is never
auto-cleared, and the vision model (when configured) is asked whether a tyre is in it at all.

This is an "unusual image" score, not a tyre detector: a product shot of a whole tyre on a white background scores
as unusual too. Only the vision model's explicit "no tyre" rejects an image (docs/evidence/subject_gate).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from treadops_ml.common import load_yaml


@lru_cache
def config() -> dict[str, Any]:
    return load_yaml("configs/subject_gate.yaml")


@lru_cache(maxsize=8)
def _reference(bundle_dir: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(Path(bundle_dir) / "reference_pca.npz")
    return z["mean"], z["components"]


def residual(emb: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Distance from the reference subspace, per row of `emb` (L2-normalised encoder features)."""
    c = np.atleast_2d(emb) - mean
    return np.linalg.norm(c - (c @ components.T) @ components, axis=1)


def assess(emb: np.ndarray, bundle_dir: Path | str) -> dict[str, Any]:
    cfg = config()
    mean, components = _reference(str(bundle_dir))
    score = float(residual(emb, mean, components)[0])
    return {"method": cfg["method"], "version": cfg["version"], "score": round(score, 4),
            "unusual_above": cfg["unusual_above"], "unusual": score > cfg["unusual_above"]}
