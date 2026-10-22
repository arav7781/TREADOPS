"""Temperature scaling on the full class-probability vector, fitted only on the calibration role."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1e-7


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    """clip -> renormalise -> softmax(log(p) / T). Works on (C,) or (N, C)."""
    if t <= 0:
        raise ValueError("temperature must be > 0")
    p = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    z = np.log(p) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out = e / e.sum(axis=1, keepdims=True)
    return out[0] if np.ndim(probs) == 1 else out


def nll(probs: np.ndarray, y_idx: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y_idx)), y_idx], EPS, 1.0)
    return float(-np.mean(np.log(p)))


@dataclass
class TemperatureCalibrator:
    temperature: float = 1.0
    version: str = "v1"
    method: str = "temperature_scaling"
    fitted_on_role: str = "calibration"
    fitted_ids_sha256: str | None = None
    n_fitted: int = 0
    nll_before: float | None = None
    nll_after: float | None = None
    preprocess_sha256: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def apply(self, raw_probs: np.ndarray) -> np.ndarray:
        return apply_temperature(raw_probs, self.temperature)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TemperatureCalibrator:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def fit_temperature(probs: np.ndarray, y_idx: np.ndarray) -> tuple[float, float, float]:
    """Return (T, nll_before, nll_after) minimising NLL over log T in [log 0.05, log 20]."""
    probs, y_idx = np.asarray(probs, float), np.asarray(y_idx, int)
    before = nll(apply_temperature(probs, 1.0), y_idx)
    res = minimize_scalar(
        lambda lt: nll(apply_temperature(probs, float(np.exp(lt))), y_idx),
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    t = float(np.exp(res.x))
    return t, before, nll(apply_temperature(probs, t), y_idx)
