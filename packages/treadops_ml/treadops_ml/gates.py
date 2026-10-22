"""Machine-readable gate report. Hard gates and quality targets are evaluated, never assumed.

status = failed               if any hard gate or quality target failed, or the comparative bound failed
       = insufficient_evidence if nothing failed but some required evidence is missing or too small
       = passed               otherwise
"""

from __future__ import annotations

from typing import Any

PASSED, FAILED, INSUFFICIENT, NA = "passed", "failed", "insufficient_evidence", "not_applicable"


def _gate(gid: str, ok: bool | None, detail: Any) -> dict[str, Any]:
    return {"id": gid, "status": INSUFFICIENT if ok is None else PASSED if ok else FAILED, "detail": detail}


def _target(gid: str, metric: dict[str, Any], target: float, min_support: int) -> dict[str, Any]:
    den = metric.get("denominator", 0)
    if not den or den < min_support or metric.get("value") is None:
        status = INSUFFICIENT
    else:
        status = PASSED if metric["value"] >= target else FAILED
    return {"id": gid, "target": target, "value": metric.get("value"), "ci95": metric.get("ci95"),
            "numerator": metric.get("numerator"), "support": den, "status": status}


def build(
    *,
    split_manifest: dict[str, Any],
    training_record: dict[str, Any],
    eval_role_ids: dict[str, set[str]],
    gate_metrics: dict[str, Any],
    parity: dict[str, Any] | None,
    comparative: dict[str, Any],
    class_mapping: dict[str, int],
    lineage: dict[str, Any],
    policy_cfg: dict[str, Any],
) -> dict[str, Any]:
    trained = {i for ids in training_record["effective_train_ids"].values() for i in ids}
    selected = {i for ids in training_record["effective_val_ids"].values() for i in ids}
    protected = set().union(*eval_role_ids.values()) if eval_role_ids else set()
    overlap = sorted((trained | selected) & protected)
    leak = split_manifest.get("leakage", {})
    hard = [
        _gate("no_critical_leakage", not any(leak.values()) and not overlap,
              {"split_checks": leak, "fit_or_selection_ids_in_protected_roles": len(overlap)}),
        _gate("lineage_complete", all(lineage.get(k) for k in (
            "git_sha", "split_sha256", "recipe_sha256", "uv_lock_sha256", "pretrained_sha256", "best_pt_sha256",
            "preprocess_sha256", "mlflow_run_id")) or None,
            {k: bool(v) for k, v in lineage.items()}),
        _gate("finite_probabilities", bool(gate_metrics.get("finite_probabilities")), "all gate-cohort outputs finite"),
        _gate("class_mapping", sorted(class_mapping) == ["defective", "good"]
              and sorted(class_mapping.values()) == [0, 1], class_mapping),
    ]
    if parity is None:
        hard.append({"id": "onnx_parity", "status": NA, "detail": "ONNX not exported"})
    else:
        hard.append(_gate("onnx_parity", parity["status"] == "passed",
                          {k: parity.get(k) for k in ("n_images", "mean_abs_probability_error",
                                                      "max_abs_probability_error", "triage_agreement")}))
    min_support = policy_cfg["comparative"]["min_gate_support_per_class"]
    b = gate_metrics["binary"]
    targets = [
        _target("defect_recall", b["defect_recall"], policy_cfg["targets"]["defect_recall"], min_support),
        _target("specificity", b["specificity"], policy_cfg["targets"]["specificity"], min_support),
    ]
    statuses = [g["status"] for g in hard + targets] + [comparative.get("status", NA)]
    if FAILED in statuses:
        overall = FAILED
    elif INSUFFICIENT in statuses:
        overall = INSUFFICIENT
    else:
        overall = PASSED
    return {
        "schema_version": 1,
        "status": overall,
        "evidence_label": gate_metrics.get("evidence_label"),
        "threshold": b["threshold"],
        "hard_gates": hard,
        "quality_targets": targets,
        "comparative": comparative,
        "triage": gate_metrics.get("triage"),
        "rule": "any failed -> failed; else any insufficient -> insufficient_evidence; else passed",
    }
