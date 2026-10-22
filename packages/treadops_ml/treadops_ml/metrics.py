"""Evaluation metrics with explicit supports and uncertainty. `defective` is always the positive class."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

INSUFFICIENT = "insufficient_evidence"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def confusion(y_def: np.ndarray, pred_def: np.ndarray) -> dict[str, int]:
    y_def, pred_def = np.asarray(y_def, bool), np.asarray(pred_def, bool)
    return {
        "tp": int((y_def & pred_def).sum()),
        "fn": int((y_def & ~pred_def).sum()),
        "tn": int((~y_def & ~pred_def).sum()),
        "fp": int((~y_def & pred_def).sum()),
    }


def _ratio(num: int, den: int) -> dict[str, Any]:
    ci = wilson(num, den)
    return {
        "value": num / den if den else None,
        "numerator": num,
        "denominator": den,
        "ci95": list(ci) if ci else None,
        "status": "ok" if den else INSUFFICIENT,
    }


def binary_metrics(p_def: Sequence[float], y_def: Sequence[bool], threshold: float) -> dict[str, Any]:
    p, y = np.asarray(p_def, float), np.asarray(y_def, bool)
    c = confusion(y, p >= threshold)
    return {
        "threshold": threshold,
        "support": {"defective": int(y.sum()), "good": int((~y).sum()), "total": int(len(y))},
        "confusion": c,
        "defect_recall": _ratio(c["tp"], c["tp"] + c["fn"]),
        "specificity": _ratio(c["tn"], c["tn"] + c["fp"]),
        "precision": _ratio(c["tp"], c["tp"] + c["fp"]),
        "accuracy": _ratio(c["tp"] + c["tn"], len(y)),
    }


def probabilistic_metrics(p_def: Sequence[float], y_def: Sequence[bool], bins: int = 10) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score

    p, y = np.clip(np.asarray(p_def, float), 1e-7, 1 - 1e-7), np.asarray(y_def, bool)
    out: dict[str, Any] = {"n": int(len(y))}
    if len(y) == 0 or y.all() or (~y).all():
        out["status"] = INSUFFICIENT
        return out
    out["average_precision"] = float(average_precision_score(y, p))
    out["brier"] = float(np.mean((p - y) ** 2))
    out["nll"] = float(-np.mean(y * np.log(p) + (~y) * np.log(1 - p)))
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rel, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, acc = float(p[m].mean()), float(y[m].mean())
            ece += m.mean() * abs(conf - acc)
            rel.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "count": int(m.sum()),
                        "mean_predicted": conf, "observed_defective_rate": acc})
    out["ece"] = float(ece)
    out["reliability"] = rel
    out["status"] = "ok"
    return out


def triage_coverage(triage: Sequence[str], y_def: Sequence[bool]) -> dict[str, Any]:
    t, y = np.asarray(triage), np.asarray(y_def, bool)
    n = len(t)
    out: dict[str, Any] = {"n": n}
    for k in ("good_candidate", "manual_review", "defect_flagged"):
        m = t == k
        out[k] = {"count": int(m.sum()), "fraction": float(m.mean()) if n else None,
                  "defective": int((m & y).sum()), "good": int((m & ~y).sum())}
    out["false_negatives_routed_to_good_candidate"] = int(((t == "good_candidate") & y).sum())
    out["defective_support"] = int(y.sum())
    return out


def latency_summary(ms: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(ms, float)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max())}


def paired_bootstrap(
    groups: Sequence[str],
    y_def: Sequence[bool],
    pred_a: Sequence[bool],
    pred_b: Sequence[bool],
    metric: str = "recall",
    resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Difference b - a in recall/specificity, resampling parents (groups) with replacement."""
    g, y = np.asarray(groups), np.asarray(y_def, bool)
    a, b = np.asarray(pred_a, bool), np.asarray(pred_b, bool)
    mask = y if metric == "recall" else ~y
    hit_a, hit_b = (a == y) & mask, (b == y) & mask
    uniq, inv = np.unique(g, return_inverse=True)
    per_den = np.bincount(inv, weights=mask, minlength=len(uniq))
    per_a = np.bincount(inv, weights=hit_a, minlength=len(uniq))
    per_b = np.bincount(inv, weights=hit_b, minlength=len(uniq))
    if per_den.sum() == 0:
        return {"status": INSUFFICIENT, "metric": metric}
    point = per_b.sum() / per_den.sum() - per_a.sum() / per_den.sum()
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(resamples):
        s = rng.integers(0, len(uniq), len(uniq))
        den = per_den[s].sum()
        if den:
            diffs.append(per_b[s].sum() / den - per_a[s].sum() / den)
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs else (None, None)
    return {"status": "ok", "metric": metric, "difference": float(point), "ci95": [float(lo), float(hi)],
            "parents": int(len(uniq)), "support": int(per_den.sum()), "resamples": resamples, "seed": seed}
