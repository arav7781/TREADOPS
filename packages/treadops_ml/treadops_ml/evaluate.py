"""Stage `evaluate`: calibration, policy selection, gate-cohort metrics, and the locked final test.

Order is fixed and each role is used for exactly one purpose:
    calibration role -> fit temperature (nothing else)
    policy role      -> select the binary threshold at the recall target
    gate cohort      -> calibration + policy (outer validation): development evidence for gates
    test role        -> run once, after everything above is frozen; every access is logged
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibrate import TemperatureCalibrator, fit_temperature
from .common import load_yaml, read_json, repo_path, sha256_file, sha256_json, write_json
from .descriptors import image_descriptors, quality_status
from .metrics import binary_metrics, latency_summary, paired_bootstrap, probabilistic_metrics, triage_coverage
from .policy import Policy, select_binary_threshold
from .preprocess import (
    PreprocessContract,
    canonical_from_raw,
    letterbox,
    load_canonical,
    tensor_from_canonical,
    to_tensor,
)
from .scenarios import derivative_bytes


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class NativeRunner:
    def __init__(self, weights: Path, device: str = "cpu", imgsz: int = 224):
        from ultralytics import YOLO

        from .bundle import tensor_input_only

        self.model = tensor_input_only(YOLO(str(weights), task="classify"))
        self.device = device
        self.contract = PreprocessContract(imgsz=imgsz)
        self.names = {int(k): v for k, v in self.model.names.items()}
        self.mapping = {v: k for k, v in self.names.items()}

    def raw(self, x: np.ndarray) -> np.ndarray:
        from .bundle import native_raw_probs

        return native_raw_probs(self.model, x, self.device, self.contract.imgsz)


def _tensors(rows: list[dict], contract: PreprocessContract) -> tuple[np.ndarray, list[dict[str, float]]]:
    xs, descs = [], []
    for r in rows:
        data = repo_path(r["path"]).read_bytes()
        xs.append(tensor_from_canonical(data, contract))
        descs.append(image_descriptors(load_canonical(data)))
    return np.concatenate(xs), descs


def _decide(policy: Policy, p: np.ndarray, descs: list[dict[str, float]]) -> list[str]:
    out = []
    for pi, d in zip(p, descs, strict=True):
        q, _ = quality_status(d, policy.quality)
        out.append(policy.decide(float(pi), q)[0])
    return out


def _cohort_report(p, y, groups, triage, threshold) -> dict[str, Any]:
    return {
        "binary": binary_metrics(p, y, threshold),
        "probabilistic": probabilistic_metrics(p, y),
        "triage": triage_coverage(triage, y),
        "parents": int(len(set(groups))),
    }


def run(
    config_path: str = "configs/evaluate.yaml",
    native_dir: str | None = None,
    out_dir: str | None = None,
    final_test: bool = True,
    champion_bundle: str | None = None,
    purpose: str = "dvc_pipeline",
    eval_dir: str | None = None,
    metrics_path: str | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    pcfg = load_yaml(cfg["policy"])
    native = repo_path(native_dir or cfg["native_dir"])
    out = repo_path(out_dir or cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    eval_root = repo_path(eval_dir or cfg["eval_dir"])
    record = read_json(native / "training_record.json")
    runner = NativeRunner(native / "best.pt", cfg.get("device", "cpu"), int(record["recipe"]["imgsz"]))
    contract = runner.contract
    dix = runner.mapping["defective"]
    rows = {r: read_jsonl(eval_root / f"{r}.jsonl") for r in ("calibration", "policy", "test")}

    def y_of(rs):
        return np.array([r["label"] == "defective" for r in rs])

    # 1) temperature on the calibration role only
    x_cal, d_cal = _tensors(rows["calibration"], contract)
    raw_cal = runner.raw(x_cal)
    y_cal_idx = np.array([runner.mapping[r["label"]] for r in rows["calibration"]])
    t, nll_b, nll_a = fit_temperature(raw_cal, y_cal_idx)
    calibrator = TemperatureCalibrator(
        temperature=t, version="v1", fitted_ids_sha256=sha256_json(sorted(r["image_id"] for r in rows["calibration"])),
        n_fitted=len(rows["calibration"]), nll_before=nll_b, nll_after=nll_a, preprocess_sha256=contract.sha256,
    )

    # 2) binary threshold on the policy role only
    x_pol, d_pol = _tensors(rows["policy"], contract)
    raw_pol = runner.raw(x_pol)
    p_pol = calibrator.apply(raw_pol)[:, dix]
    sel = select_binary_threshold(p_pol, y_of(rows["policy"]), pcfg["targets"]["defect_recall"])
    policy = Policy.from_config(pcfg)
    if sel.get("threshold") is not None:
        policy.binary_threshold = float(sel["threshold"])
    policy.selection = {**sel, "selected_on_role": "policy",
                        "policy_ids_sha256": sha256_json(sorted(r["image_id"] for r in rows["policy"]))}

    # 3) gate cohort = outer validation (development evidence)
    p_cal = calibrator.apply(raw_cal)[:, dix]
    gate_rows = rows["calibration"] + rows["policy"]
    p_gate = np.concatenate([p_cal, p_pol])
    y_gate = y_of(gate_rows)
    tri_gate = _decide(policy, p_gate, d_cal + d_pol)
    gate = _cohort_report(p_gate, y_gate, [r["group_id"] for r in gate_rows], tri_gate, policy.binary_threshold)
    gate["by_role"] = {
        "calibration": binary_metrics(p_cal, y_of(rows["calibration"]), policy.binary_threshold),
        "policy": binary_metrics(p_pol, y_of(rows["policy"]), policy.binary_threshold),
    }
    gate["evidence_label"] = "development evidence (outer validation: calibration + policy roles)"
    gate["finite_probabilities"] = bool(np.isfinite(raw_cal).all() and np.isfinite(raw_pol).all())
    gate_preds = pd.DataFrame({
        "image_id": [r["image_id"] for r in gate_rows], "group_id": [r["group_id"] for r in gate_rows],
        "role": [r["role"] for r in gate_rows], "label": [r["label"] for r in gate_rows],
        "raw_defective": np.concatenate([raw_cal[:, dix], raw_pol[:, dix]]), "p_defective": p_gate,
        "triage": tri_gate,
    })
    gate_preds.to_parquet(out / "gate_predictions.parquet", index=False)

    comparative: dict[str, Any] = {"status": "not_applicable", "reason": "no champion supplied"}
    if champion_bundle:
        comparative = compare_with_champion(champion_bundle, gate_rows, p_gate, y_gate, policy, pcfg)

    # 4) locked final test (once, logged)
    test_report: dict[str, Any] = {"status": "not_run"}
    stress: dict[str, Any] = {"status": "not_run"}
    if final_test:
        x_test, d_test = _tensors(rows["test"], contract)
        p_test = calibrator.apply(runner.raw(x_test))[:, dix]
        y_test = y_of(rows["test"])
        g_test = [r["group_id"] for r in rows["test"]]
        tri_test = _decide(policy, p_test, d_test)
        test_report = _cohort_report(p_test, y_test, g_test, tri_test, policy.binary_threshold)
        test_report["status"] = "ok"
        test_report["evidence_label"] = "locked final test (frozen checkpoint, calibrator and policy)"
        stress = stress_eval(runner, calibrator, policy, rows["test"], p_test, tri_test, cfg)
        log = repo_path(cfg["test_access_log"])
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as fh:
            fh.write(json.dumps({
                "at": datetime.now(UTC).isoformat(), "purpose": purpose,
                "best_pt_sha256": record["best_pt_sha256"], "policy": policy.version,
                "calibration_temperature": round(t, 6), "n_test": len(rows["test"]),
            }) + "\n")

    latency = measure_latency(runner, rows["policy"][:60], contract)

    write_json(out / "calibration.json", calibrator.to_dict())
    write_json(out / "policy.json", policy.to_dict())
    write_json(out / "gate_metrics.json", gate)
    write_json(out / "comparative.json", comparative)
    write_json(out / "test_metrics.json", test_report)
    write_json(out / "stress.json", stress)
    write_json(out / "latency.json", latency)
    eval_record = {
        "best_pt_sha256": sha256_file(native / "best.pt"),
        "class_names": runner.names,
        "role_hashes": {r: sha256_json(sorted(x["image_id"] for x in rows[r])) for r in rows},
        "role_counts": {r: len(rows[r]) for r in rows},
        "preprocess_sha256": contract.sha256,
        "final_test_run": final_test,
    }
    write_json(out / "evaluation_record.json", eval_record)

    def v(m):
        return m["value"] if m and m.get("value") is not None else None

    summary = {
        "calibration_temperature": t,
        "binary_threshold": policy.binary_threshold,
        "gate_defect_recall": v(gate["binary"]["defect_recall"]),
        "gate_specificity": v(gate["binary"]["specificity"]),
        "gate_average_precision": gate["probabilistic"].get("average_precision"),
        "test_defect_recall": v(test_report.get("binary", {}).get("defect_recall")),
        "test_specificity": v(test_report.get("binary", {}).get("specificity")),
        "test_average_precision": test_report.get("probabilistic", {}).get("average_precision"),
        "test_brier": test_report.get("probabilistic", {}).get("brier"),
        "model_p95_ms_cpu": latency["model_only"].get("p95_ms"),
    }
    write_json(metrics_path or cfg["metrics_path"], summary)
    return summary


def stress_eval(runner, calibrator, policy, test_rows, p_clean, tri_clean, cfg) -> dict[str, Any]:
    dix = runner.mapping["defective"]
    y = np.array([r["label"] == "defective" for r in test_rows])
    groups = [r["group_id"] for r in test_rows]
    clean_pred = p_clean >= policy.binary_threshold
    out: dict[str, Any] = {"parents": len(set(groups)), "pairing": "same test parents, derivative per scenario",
                           "clean": binary_metrics(p_clean, y, policy.binary_threshold),
                           "clean_triage": triage_coverage(tri_clean, y), "scenarios": {}}
    for name, transform in cfg["stress"]["scenarios"].items():
        xs, descs = [], []
        for r in test_rows:
            rgb = load_canonical(repo_path(r["path"]).read_bytes())
            canonical, _ = canonical_from_raw(derivative_bytes(rgb, transform), runner.contract)
            crgb = load_canonical(canonical)
            xs.append(to_tensor(letterbox(crgb, runner.contract)[0], runner.contract))
            descs.append(image_descriptors(crgb))
        p = calibrator.apply(runner.raw(np.concatenate(xs)))[:, dix]
        tri = _decide(policy, p, descs)
        out["scenarios"][name] = {
            "transform": transform,
            "binary": binary_metrics(p, y, policy.binary_threshold),
            "triage": triage_coverage(tri, y),
            "recall_vs_clean": paired_bootstrap(groups, y, clean_pred, p >= policy.binary_threshold, "recall",
                                                cfg["bootstrap"]["resamples"], cfg["bootstrap"]["seed"]),
            "specificity_vs_clean": paired_bootstrap(groups, y, clean_pred, p >= policy.binary_threshold,
                                                     "specificity", cfg["bootstrap"]["resamples"],
                                                     cfg["bootstrap"]["seed"]),
            "quality_low_fraction": float(np.mean([quality_status(d, policy.quality)[0] != "accepted"
                                                   for d in descs])),
        }
    return out


def compare_with_champion(champion_dir, gate_rows, p_gate, y_gate, policy, pcfg) -> dict[str, Any]:
    from .bundle import Bundle

    champ = Bundle(champion_dir, device="cpu")
    xs = np.concatenate([tensor_from_canonical(repo_path(r["path"]).read_bytes(), champ.preprocess)
                         for r in gate_rows])
    p_champ = champ.p_defective(champ.raw_probs(xs))
    groups = [r["group_id"] for r in gate_rows]
    pred_c = p_champ >= champ.policy.binary_threshold
    pred_n = p_gate >= policy.binary_threshold
    diff = paired_bootstrap(groups, y_gate, pred_c, pred_n, "recall")
    spec = paired_bootstrap(groups, y_gate, pred_c, pred_n, "specificity")
    min_support = pcfg["comparative"]["min_gate_support_per_class"]
    max_drop = pcfg["comparative"]["max_recall_drop_pp"] / 100
    if int(y_gate.sum()) < min_support or int((~y_gate).sum()) < min_support or diff["status"] != "ok":
        status = "insufficient_evidence"
    else:
        status = "passed" if diff["difference"] >= -max_drop else "failed"
    return {"status": status, "champion_bundle_sha256": champ.sha, "recall_difference": diff,
            "specificity_difference": spec, "max_recall_drop_pp": pcfg["comparative"]["max_recall_drop_pp"],
            "rule": "candidate recall - champion recall >= -max_drop (point estimate) with CI reported"}


def measure_latency(runner, rows, contract) -> dict[str, Any]:
    pre, model, total = [], [], []
    for i, r in enumerate(rows):
        data = repo_path(r["path"]).read_bytes()
        t0 = time.perf_counter()
        x = tensor_from_canonical(data, contract)
        t1 = time.perf_counter()
        runner.raw(x)
        t2 = time.perf_counter()
        if i >= 5:  # warm-up excluded
            pre.append((t1 - t0) * 1000)
            model.append((t2 - t1) * 1000)
            total.append((t2 - t0) * 1000)
    from .common import hardware

    return {"device": runner.device, "warmup_excluded": 5, "batch": 1,
            "preprocess": latency_summary(pre), "model_only": latency_summary(model),
            "preprocess_plus_model": latency_summary(total), "hardware": hardware()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="calibrate, select policy, evaluate gates and the locked test")
    ap.add_argument("--config", default="configs/evaluate.yaml")
    ap.add_argument("--native-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-final-test", action="store_true")
    ap.add_argument("--champion-bundle", default=None)
    args = ap.parse_args(argv)
    print(json.dumps(run(args.config, args.native_dir, args.out_dir, not args.no_final_test,
                         args.champion_bundle), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
