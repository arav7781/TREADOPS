"""Stage `export_parity`: ONNX export and native-vs-ONNX parity on held-out development images.

Both runtimes receive the identical prepared tensor. Tolerances come from configs/policy.yaml and are never
widened to hide a defect; a failure is reported and the ONNX file is left out of the bundle.
"""

from __future__ import annotations

import argparse
import json
import shutil
from typing import Any

import numpy as np

from . import tvmodel
from .calibrate import TemperatureCalibrator
from .common import load_yaml, read_json, repo_path, sha256_file, write_json
from .evaluate import NativeRunner, read_jsonl
from .policy import Policy
from .preprocess import tensor_from_canonical


def run(native_dir: str = "artifacts/native", evaluation_dir: str = "artifacts/evaluation",
        out_dir: str = "artifacts/export", eval_dir: str = "data/eval") -> dict[str, Any]:
    from ultralytics import YOLO

    native, evaluation, out = repo_path(native_dir), repo_path(evaluation_dir), repo_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pcfg = load_yaml("configs/policy.yaml")["parity"]
    record = read_json(native / "training_record.json")
    imgsz = int(record["recipe"]["imgsz"])

    runner = NativeRunner(native / "best.pt", "cpu", imgsz)
    onnx_path = out / "model.onnx"
    if runner.family == tvmodel.FAMILY:
        tvmodel.export_onnx(runner.model, onnx_path, imgsz)
    else:
        work = out / "work"
        work.mkdir(exist_ok=True)
        shutil.copy2(native / "best.pt", work / "best.pt")
        exported = YOLO(str(work / "best.pt"), task="classify").export(
            format="onnx", imgsz=imgsz, dynamic=True, simplify=True, device="cpu", verbose=False)
        shutil.move(exported, onnx_path)
        shutil.rmtree(work)

    rows = read_jsonl(repo_path(eval_dir) / "policy.jsonl") + read_jsonl(repo_path(eval_dir) / "calibration.jsonl")
    rows = rows[: max(pcfg["min_images"], 120)]
    x = np.concatenate([tensor_from_canonical(repo_path(r["path"]).read_bytes(), runner.contract) for r in rows])
    native_raw = runner.raw(x)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    onnx_raw = np.concatenate([sess.run(None, {inp: x[i : i + 32]})[0] for i in range(0, len(x), 32)]).astype(
        np.float64)

    err = np.abs(native_raw - onnx_raw)
    cal = TemperatureCalibrator.from_dict(read_json(evaluation / "calibration.json"))
    pol = Policy.from_dict(read_json(evaluation / "policy.json"))
    dix = runner.mapping["defective"]
    p_n, p_o = cal.apply(native_raw)[:, dix], cal.apply(onnx_raw)[:, dix]
    tri_n = [pol.decide(float(p))[0] for p in p_n]
    tri_o = [pol.decide(float(p))[0] for p in p_o]
    band = pcfg["triage_boundary_band"]
    mismatches = []
    for i, (a, b) in enumerate(zip(tri_n, tri_o, strict=True)):
        if a != b:
            near = min(abs(p_n[i] - pol.triage_low), abs(p_n[i] - pol.triage_high)) <= band
            mismatches.append({"image_id": rows[i]["image_id"], "native": a, "onnx": b,
                               "p_native": float(p_n[i]), "p_onnx": float(p_o[i]), "inside_boundary_band": near})
    hard = [m for m in mismatches if not m["inside_boundary_band"]]
    passed = (len(rows) >= pcfg["min_images"] and float(err.mean()) <= pcfg["fp32_mean_abs_probability_error"]
              and float(err.max()) <= pcfg["fp32_max_abs_probability_error"] and not hard)
    report = {
        "status": "passed" if passed else "failed",
        "n_images": len(rows),
        "cohort": "held-out development images (policy + calibration roles)",
        "precision": "fp32",
        "runtimes": {"native": "torch cpu (torchvision)" if runner.family == tvmodel.FAMILY else "ultralytics/torch cpu",
                     "onnx": f"onnxruntime {ort.__version__} CPUExecutionProvider"},
        "mean_abs_probability_error": float(err.mean()),
        "max_abs_probability_error": float(err.max()),
        "tolerances": {"mean": pcfg["fp32_mean_abs_probability_error"], "max": pcfg["fp32_max_abs_probability_error"]},
        "triage_agreement": float(np.mean(np.array(tri_n) == np.array(tri_o))),
        "triage_mismatches": mismatches,
        "triage_boundary_band": band,
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_input": {"name": inp, "shape": [str(d) for d in sess.get_inputs()[0].shape]},
        "best_pt_sha256": sha256_file(native / "best.pt"),
    }
    write_json(out / "parity.json", report)
    return {k: report[k] for k in ("status", "n_images", "mean_abs_probability_error", "max_abs_probability_error",
                                   "triage_agreement")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ONNX export + native parity")
    ap.add_argument("--native-dir", default="artifacts/native")
    ap.add_argument("--evaluation-dir", default="artifacts/evaluation")
    ap.add_argument("--out-dir", default="artifacts/export")
    args = ap.parse_args(argv)
    print(json.dumps(run(args.native_dir, args.evaluation_dir, args.out_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
