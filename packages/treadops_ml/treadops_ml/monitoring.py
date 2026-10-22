"""Drift statistics. A distribution change is an investigation signal, never an automatic model decision."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EVIDENTLY_VERSION = "0.7.23"


def frozen_bins(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile edges from the reference cohort only; outer edges are open so every value lands in a bin."""
    qs = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.concatenate([[-np.inf], qs, [np.inf]])


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10, smoothing: float = 1e-4) -> dict[str, Any]:
    edges = frozen_bins(np.asarray(reference, float), n_bins)
    r = np.histogram(reference, edges)[0].astype(float)
    c = np.histogram(current, edges)[0].astype(float)
    r = r / r.sum() + smoothing
    c = c / max(c.sum(), 1) + smoothing
    r, c = r / r.sum(), c / c.sum()
    value = float(np.sum((c - r) * np.log(c / r)))
    return {"psi": value, "bins": len(edges) - 1, "smoothing": smoothing,
            "reference_mean": float(np.mean(reference)), "current_mean": float(np.mean(current)),
            "current_distribution": c.round(4).tolist(), "reference_distribution": r.round(4).tolist()}


def domain_classifier_auc(reference: np.ndarray, current: np.ndarray, seed: int = 42) -> dict[str, Any]:
    """Held-out (5-fold) logistic-regression AUC separating reference from current embeddings. 0.5 = no shift."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    x = np.vstack([reference, current])
    y = np.concatenate([np.zeros(len(reference)), np.ones(len(current))])
    if min(len(reference), len(current)) < 10:
        return {"status": "insufficient_data", "auc": None}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), x, y, cv=cv, method="predict_proba")[:, 1]
    return {"status": "ok", "auc": float(roc_auc_score(y, proba)), "folds": 5, "seed": seed,
            "n_reference": int(len(reference)), "n_current": int(len(current)),
            "note": "diagnostic only; not a defect-recall estimate"}


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """Version-pinned Evidently DataDriftPreset adapter. Returns (summary, full json) or an explicit error."""
    try:
        import evidently
        from evidently import Report
        from evidently.presets import DataDriftPreset

        if evidently.__version__ != EVIDENTLY_VERSION:
            return {"status": "skipped", "reason": f"evidently {evidently.__version__} != pinned {EVIDENTLY_VERSION}"}
        snap = Report([DataDriftPreset(columns=columns)]).run(current_data=current[columns],
                                                               reference_data=reference[columns])
        full = snap.dict()
        drifted = None
        per_column = {}
        for m in full.get("metrics", []):
            name = str(m.get("metric_name", ""))
            val = m.get("value")
            if name.startswith("DriftedColumnsCount") and isinstance(val, dict):
                drifted = val
            elif name.startswith("ValueDrift") and isinstance(val, int | float):
                col = name.split("column=")[-1].split(",")[0].rstrip(")")
                per_column[col] = float(val)
        return {"status": "ok", "version": evidently.__version__, "drifted_columns": drifted,
                "per_column_drift_score": per_column, "full": full}
    except Exception as exc:  # the monitor must not fail because the report library failed
        return {"status": "error", "error": f"{exc.__class__.__name__}: {str(exc)[:300]}"}
