"""Versioned decision policy: a binary evaluation threshold kept separate from three-way triage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

TRIAGES = ("good_candidate", "manual_review", "defect_flagged")


@dataclass
class Policy:
    version: str = "v1"
    positive_label: str = "defective"
    binary_threshold: float = 0.5
    triage_low: float = 0.2
    triage_high: float = 0.8
    final_clearance: str = "human"
    quality: dict[str, float] = field(default_factory=lambda: {
        "min_brightness": 0.12, "max_brightness": 0.95, "min_blur_variance": 15.0})
    selection: dict[str, Any] = field(default_factory=dict)

    def decide(self, p_defective: float, quality_status: str = "accepted") -> tuple[str, str]:
        """Return (triage, reason). A low-quality photo can never be auto-routed to good_candidate."""
        if not np.isfinite(p_defective):
            raise ValueError("non-finite probability")
        if p_defective >= self.triage_high:
            return "defect_flagged", f"p_defective {p_defective:.3f} >= {self.triage_high}"
        if p_defective < self.triage_low:
            if quality_status != "accepted":
                return "manual_review", "quality escalation: low-quality image cannot be auto-cleared"
            return "good_candidate", f"p_defective {p_defective:.3f} < {self.triage_low}"
        return "manual_review", f"{self.triage_low} <= p_defective {p_defective:.3f} < {self.triage_high}"

    def binary(self, p_defective: float) -> bool:
        return p_defective >= self.binary_threshold

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Policy:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Policy:
        return cls(
            version=cfg["version"],
            positive_label=cfg["positive_label"],
            binary_threshold=cfg["binary_threshold"],
            triage_low=cfg["triage"]["low"],
            triage_high=cfg["triage"]["high"],
            final_clearance=cfg["triage"]["final_clearance"],
            quality=dict(cfg["quality"]),
        )


def select_binary_threshold(p_def: np.ndarray, y_def: np.ndarray, recall_target: float) -> dict[str, Any]:
    """Highest threshold whose recall on this cohort meets the target (maximises specificity)."""
    p, y = np.asarray(p_def, float), np.asarray(y_def, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return {"status": "insufficient_evidence", "threshold": None}
    best = None
    for t in sorted(set(np.round(p, 6).tolist()) | {0.5}, reverse=True):
        recall = float((p[y] >= t).mean())
        if recall >= recall_target:
            spec = float((p[~y] < t).mean())
            best = {"threshold": float(t), "recall": recall, "specificity": spec}
            break
    if best is None:
        return {"status": "target_unreachable", "threshold": float(p.min())}
    return {"status": "ok", **best, "n": int(len(y)), "rule": "max_specificity_at_recall_target",
            "recall_target": recall_target}
