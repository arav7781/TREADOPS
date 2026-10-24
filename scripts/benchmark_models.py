"""Benchmark candidate classifiers on the multi-source split and pick one, on development evidence only.

Every candidate trains on dev_train (checkpoint chosen on dev_val), gets its temperature from the calibration role
and its threshold from the policy role, exactly like the `evaluate` stage. It is then scored on the gate cohort
(calibration + policy), per source, and under low light and blur. The locked test is NOT used here: the winner gets
the full `evaluate` stage, which runs the locked test once.

Selection rule (fixed before any result was seen):
  1. must meet the gate targets on the gate cohort (defect recall >= 0.95, specificity >= 0.90);
  2. among those, the highest worst-source defect recall (it has to work on every camera and place), over the
     sources with at least MIN_SOURCE_DEFECTIVE defective gate-cohort photos (smaller ones are reported only);
  3. ties (within 0.005): higher low-light recall, then lower CPU latency.

Candidates: YOLO26 n/s/m with the robustness augmentation, YOLO26 s without it (control), and ConvNeXt-Tiny
(torchvision, ImageNet-1k) trained on the same images with the same transforms. Every training run is logged to
MLflow with tag purpose=benchmark. Writes docs/evidence/benchmark/<timestamp>.json.

Usage: python scripts/benchmark_models.py [candidate ...]     (default: all)
       python scripts/benchmark_models.py --adopt <candidate>  (put the winner in artifacts/native)
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from treadops_ml.calibrate import TemperatureCalibrator, fit_temperature
from treadops_ml.common import load_yaml, repo_path, sha256_file
from treadops_ml.descriptors import image_descriptors
from treadops_ml.evaluate import NativeRunner, _by_source, _decide, read_jsonl
from treadops_ml.metrics import binary_metrics, probabilistic_metrics
from treadops_ml.policy import Policy, select_binary_threshold
from treadops_ml.preprocess import (
    PreprocessContract,
    canonical_from_raw,
    letterbox,
    load_canonical,
    to_tensor,
)
from treadops_ml.scenarios import derivative_bytes
from treadops_ml.weights import pinned_sha, pretrained

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/benchmark"
CANDIDATES = {
    "yolo26n_robust": {"kind": "yolo", "recipe": "configs/recipes/nano_224_robust.yaml"},
    "yolo26s_robust": {"kind": "yolo", "recipe": "configs/recipes/small_224_robust.yaml"},
    "yolo26m_robust": {"kind": "yolo", "recipe": "configs/recipes/medium_224_robust.yaml"},
    "yolo26s_control": {"kind": "yolo", "recipe": "configs/recipes/small_224.yaml"},
    "convnext_tiny_robust": {"kind": "torchvision", "arch": "convnext_tiny", "epochs": 30, "lr": 1e-4, "batch": 32,
                             "patience": 6, "seed": 42, "augment": "configs/recipes/small_224_robust.yaml",
                             "weights": "convnext_tiny-983f1562.pth"},
}
SCENARIOS = {"low_light": {"brightness_factor": 0.45}, "camera_blur": {"gaussian_sigma": 1.5}}
MIN_SOURCE_DEFECTIVE = 30  # MSU has 1 defective photo in the gate cohort: a recall of 0 or 1 says nothing
CLASSES = ["defective", "good"]  # ImageFolder order; YOLO reads the same folders


# ---- ConvNeXt-Tiny through torchvision, same canonical bytes, letterbox and augmentation ---------------------------
def _normalised(net):
    import torch
    from torch import nn

    class Normalised(nn.Module):
        """Takes the same [0, 1] letterboxed tensor as YOLO and applies ImageNet normalisation inside."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            return self.inner((x - self.mean) / self.std)

    return Normalised(net)


def _folder_samples(split: str) -> list[tuple[Path, int]]:
    root = repo_path("data/yolo/dev") / split
    return [(p, i) for i, c in enumerate(CLASSES) for p in sorted((root / c).glob("*.jpg"))]


class _Folder:
    """(path, class) samples -> (CHW tensor, class); loaded and augmented inside DataLoader workers."""

    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, cls = self.samples[i]
        return self.transform(load_canonical(path.read_bytes())), cls


def train_torchvision(name: str, spec: dict, out: Path) -> dict:
    import mlflow
    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import DataLoader

    from treadops_ml.transforms import AugmentConfig, EvalTransform, TrainTransform

    random.seed(spec["seed"])
    np.random.seed(spec["seed"])
    torch.manual_seed(spec["seed"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    contract = PreprocessContract(imgsz=224)
    aug = AugmentConfig(**load_yaml(spec["augment"])["augment"])
    train_t, eval_t = TrainTransform(contract, aug), EvalTransform(contract)
    train, val = _folder_samples("train"), _folder_samples("val")
    net = torchvision.models.convnext_tiny()  # ImageNet-1k weights, SHA-256 pinned in configs/weights.lock.yaml
    net.load_state_dict(torch.load(pretrained(spec["weights"]), map_location="cpu", weights_only=True))
    net.classifier[2] = nn.Linear(net.classifier[2].in_features, len(CLASSES))
    model = _normalised(net).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=spec["epochs"])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Workers are seeded from the generator (python `random`, used by TrainTransform, included), so runs repeat.
    loaders = {k: DataLoader(_Folder(samples, t), batch_size=spec["batch"], shuffle=k == "train", num_workers=4,
                             persistent_workers=True, generator=torch.Generator().manual_seed(spec["seed"]))
               for k, samples, t in (("train", train, train_t), ("val", val, eval_t))}

    def batches(split):
        for x, y in loaders[split]:
            yield x.to(device), y.to(device)

    mlflow.set_experiment("treadops-yolo-classify")
    run = mlflow.start_run(run_name=name, tags={"purpose": "benchmark", "model_family": "convnext_tiny",
                                                "split_sha": json.loads((ROOT / "data/manifests/split_manifest.json")
                                                                        .read_text())["split_sha256"]})
    mlflow.log_params({"arch": spec["arch"], "epochs": spec["epochs"], "lr": spec["lr"], "batch": spec["batch"],
                       "pretrained": spec["weights"], "pretrained_sha256": pinned_sha(spec["weights"]),
                       **{f"augment.{k}": v for k, v in vars(aug).items()}})
    best, bad, t0, history = float("inf"), 0, time.time(), []
    for epoch in range(1, spec["epochs"] + 1):
        model.train()
        tl, n = 0.0, 0
        for x, y in batches("train"):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tl, n = tl + loss.item() * len(y), n + len(y)
        sched.step()
        model.eval()
        vl, correct, m = 0.0, 0, 0
        with torch.inference_mode():
            for x, y in batches("val"):
                logits = model(x)
                vl, correct, m = vl + float(loss_fn(logits, y)) * len(y), correct + int((logits.argmax(1) == y).sum()), m + len(y)
        row = {"epoch": epoch, "train_loss": tl / n, "val_loss": vl / m, "val_accuracy": correct / m}
        history.append(row)
        mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        print(f"[{name}] epoch {epoch}: {row}", flush=True)
        if row["val_loss"] < best - 1e-4:
            best, bad = row["val_loss"], 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= spec["patience"]:
                break
    mlflow.log_artifact(str(out / "best.pt"), artifact_path="weights")
    mlflow.end_run()
    return {"duration_s": round(time.time() - t0, 1), "epochs_completed": len(history), "history": history,
            "best_pt_sha256": sha256_file(out / "best.pt"), "mlflow_run_id": run.info.run_id, "device": device}


class TorchvisionRunner:
    """Serves through ONNX Runtime, as a torchvision bundle would (PyTorch's CPU path is ~9x slower for ConvNeXt)."""

    def __init__(self, weights: Path):
        import torch
        import torchvision
        from torch import nn

        net = torchvision.models.convnext_tiny(weights=None)
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, len(CLASSES))
        self.model = _normalised(net)
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()
        self.contract = PreprocessContract(imgsz=224)
        self.mapping = {c: i for i, c in enumerate(CLASSES)}
        import onnxruntime as ort

        onnx = Path(weights).with_suffix(".onnx")
        torch.onnx.export(self.model, torch.zeros(1, 3, 224, 224), str(onnx), input_names=["images"],
                          output_names=["logits"], dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17, dynamo=False)
        self.session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])

    def raw(self, x: np.ndarray) -> np.ndarray:
        z = self.session.run(None, {"images": x.astype(np.float32)})[0]
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)


# ---- scoring on development evidence ----------------------------------------------------------------------------------
def _tensors(rows, contract, scenario=None):
    xs, descs = [], []
    for r in rows:
        rgb = load_canonical(repo_path(r["path"]).read_bytes())
        if scenario:
            canonical, _ = canonical_from_raw(derivative_bytes(rgb, scenario), contract)
            rgb = load_canonical(canonical)
        xs.append(to_tensor(letterbox(rgb, contract)[0], contract))
        descs.append(image_descriptors(rgb))
    return np.concatenate(xs), descs


def _batched(runner, x, size=64):
    return np.concatenate([runner.raw(x[i : i + size]) for i in range(0, len(x), size)])


def score(runner) -> dict:
    pcfg = load_yaml("configs/policy.yaml")
    rows = {r: read_jsonl(repo_path("data/eval") / f"{r}.jsonl") for r in ("calibration", "policy")}
    dix = runner.mapping["defective"]
    y = {r: np.array([x["label"] == "defective" for x in rows[r]]) for r in rows}
    x_cal, d_cal = _tensors(rows["calibration"], runner.contract)
    x_pol, d_pol = _tensors(rows["policy"], runner.contract)
    raw_cal, raw_pol = _batched(runner, x_cal), _batched(runner, x_pol)
    t, *_ = fit_temperature(raw_cal, np.array([runner.mapping[x["label"]] for x in rows["calibration"]]))
    cal = TemperatureCalibrator(temperature=t, version="bench", fitted_ids_sha256="", n_fitted=len(raw_cal),
                                nll_before=0.0, nll_after=0.0, preprocess_sha256=runner.contract.sha256)
    p_pol = cal.apply(raw_pol)[:, dix]
    sel = select_binary_threshold(p_pol, y["policy"], pcfg["targets"]["defect_recall"])
    policy = Policy.from_config(pcfg)
    if sel.get("threshold") is not None:
        policy.binary_threshold = float(sel["threshold"])
    gate_rows = rows["calibration"] + rows["policy"]
    p_gate = np.concatenate([cal.apply(raw_cal)[:, dix], p_pol])
    y_gate = np.concatenate([y["calibration"], y["policy"]])
    tri = _decide(policy, p_gate, d_cal + d_pol)
    out = {"temperature": t, "threshold": policy.binary_threshold,
           "gate": binary_metrics(p_gate, y_gate, policy.binary_threshold),
           "average_precision": probabilistic_metrics(p_gate, y_gate).get("average_precision"),
           "by_source": {s: {"defect_recall": r["binary"]["defect_recall"]["value"],
                             "specificity": r["binary"]["specificity"]["value"], "parents": r["parents"],
                             "defective": r["binary"]["defect_recall"]["denominator"]}
                         for s, r in _by_source(gate_rows, p_gate, y_gate, tri, policy.binary_threshold).items()},
           "stress": {}}
    for name, transform in SCENARIOS.items():
        xs, _ = _tensors(gate_rows, runner.contract, transform)
        p = cal.apply(_batched(runner, xs))[:, dix]
        out["stress"][name] = binary_metrics(p, y_gate, policy.binary_threshold)
    lat = []
    for i in range(min(60, len(x_pol))):
        t0 = time.perf_counter()
        runner.raw(x_pol[i : i + 1])
        lat.append((time.perf_counter() - t0) * 1000)
    out["cpu_p95_ms"] = float(np.percentile(lat[5:], 95))
    return out


def value(m):
    return (m or {}).get("value")


def choose(results: dict) -> dict:
    ok = {k: r for k, r in results.items() if "score" in r and (value(r["score"]["gate"]["defect_recall"]) or 0) >= 0.95
          and (value(r["score"]["gate"]["specificity"]) or 0) >= 0.90}
    if not ok:
        return {"winner": None, "reason": "no candidate met the gate targets on the gate cohort"}

    def key(item):
        r = item[1]["score"]
        worst = min(s["defect_recall"] for s in r["by_source"].values()
                    if s["defect_recall"] is not None and s["defective"] >= MIN_SOURCE_DEFECTIVE)
        return (round(worst / 0.005), value(r["stress"]["low_light"]["defect_recall"]) or 0, -r["cpu_p95_ms"])

    winner = max(ok.items(), key=key)[0]
    return {"winner": winner, "eligible": sorted(ok), "rule": __doc__.split("Selection rule")[1].split("Candidates")[0].strip()}


ADOPT_RECIPES = {"convnext_tiny_robust": "configs/recipes/convnext_tiny_224_robust.yaml"}


def adopt(name: str, out_dir: str = "artifacts/native") -> dict:
    """Put a benchmarked candidate where the pipeline's `train` stage leaves its model, without retraining it:
    the checkpoint in the model family's own format plus a full training record that says where it came from.
    Only torchvision candidates need this (a YOLO winner is trained by the `train` stage with its recipe)."""
    import torch

    from treadops_ml import tvmodel
    from treadops_ml.common import git_state, hardware, lock_hash, package_versions, sha256_json, write_json
    from treadops_ml.train import _folder_ids

    res = json.loads((OUT / "results.json").read_text())[name]
    split = json.loads((ROOT / "data/manifests/split_manifest.json").read_text())
    if res["split_sha256"] != split["split_sha256"]:
        raise SystemExit(f"{name} was benchmarked on another split; retrain it with the `train` stage instead")
    rec = load_yaml(ADOPT_RECIPES[name])
    out = repo_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(OUT / name / "best.pt", map_location="cpu", weights_only=True)
    torch.save({"family": tvmodel.FAMILY, "arch": rec["arch"], "class_names": dict(enumerate(tvmodel.CLASSES)),
                "state_dict": {k.removeprefix("inner."): v for k, v in state.items() if k.startswith("inner.")}},
               out / "best.pt")
    names = tvmodel.load(out / "best.pt")[1]
    ids = _folder_ids(repo_path("data/yolo/dev"))
    contract = PreprocessContract(imgsz=int(rec["imgsz"]))
    fitted = sum(len(v) for v in ids["train"].values())
    epochs = [{"epoch": h["epoch"], "train_loss": h["train_loss"], "val/loss": h["val_loss"],
               "metrics/accuracy_top1": h["val_accuracy"]} for h in res["train"]["history"]]
    record = {
        "schema_version": 1, "run_name": name, "status": "completed", "recipe": rec, "recipe_sha256": sha256_json(rec),
        "train_args": {**{k: rec[k] for k in ("arch", "epochs", "imgsz", "batch", "lr0", "weight_decay",
                                              "label_smoothing", "patience", "seed")},
                       "optimizer": "AdamW", "scheduler": "cosine", "checkpoint_rule": "lowest val loss",
                       "device": res["train"]["device"]},
        "device": res["train"]["device"], "duration_s": res["train"]["duration_s"],
        "epochs_completed": res["train"]["epochs_completed"], "epoch_metrics": epochs, "final_best_pt_validation": {},
        "class_names": names, "defective_index": next(k for k, v in names.items() if v == "defective"),
        "pretrained": {"name": rec["model"], "sha256": pinned_sha(rec["model"])},
        "best_pt_sha256": sha256_file(out / "best.pt"), "data_root": "data/yolo/dev",
        "effective_train_ids": ids["train"], "effective_val_ids": ids["val"], "fitted_image_count": fitted,
        "train_pool_image_count": fitted + sum(len(v) for v in ids["val"].values()),
        "split_sha256": split["split_sha256"], "preprocess": contract.to_dict(), "preprocess_sha256": contract.sha256,
        "uv_lock_sha256": lock_hash(), "packages": package_versions(), "hardware": hardware(),
        "nondeterminism_note": "seeded; bitwise GPU/MPS reproducibility is not guaranteed",
        "adopted_from": {"benchmark": name, "trained_by": "scripts/benchmark_models.py (same split, transforms and "
                         f"hyperparameters as {ADOPT_RECIPES[name]})",
                         "benchmark_best_pt_sha256": res["train"]["best_pt_sha256"],
                         "adopted_at": datetime.now(UTC).isoformat()},
        **git_state(), "mlflow_run_id": res["train"]["mlflow_run_id"],
    }
    write_json(out / "training_record.json", record)
    with (out / "epochs.jsonl").open("w") as fh:
        for e in epochs:
            fh.write(json.dumps(e) + "\n")
    return {k: record[k] for k in ("run_name", "best_pt_sha256", "epochs_completed", "fitted_image_count",
                                   "mlflow_run_id")}


def main(names: list[str]) -> int:
    import mlflow

    from treadops_ml import train

    split = json.loads((ROOT / "data/manifests/split_manifest.json").read_text())
    results_path = OUT / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name in names:
        spec = CANDIDATES[name]
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        if results.get(name, {}).get("split_sha256") != split["split_sha256"] or "score" not in results.get(name, {}):
            print(f"== training {name}", flush=True)
            if spec["kind"] == "yolo":
                rec = train.run(recipe=spec["recipe"], out_dir=str(out), name=f"bench_{name}",
                                extra_tags={"purpose": "benchmark"})
                info = {k: rec[k] for k in ("duration_s", "epochs_completed", "best_pt_sha256", "mlflow_run_id", "device")}
                runner = NativeRunner(out / "best.pt", "cpu", 224)
            else:
                info = train_torchvision(name, spec, out)
                runner = TorchvisionRunner(out / "best.pt")
            print(f"== scoring {name}", flush=True)
            results[name] = {"spec": spec, "split_sha256": split["split_sha256"], "train": info, "score": score(runner)}
            results_path.write_text(json.dumps(results, indent=2, default=float))
    decision = choose(results)
    report = {"at": datetime.now(UTC).isoformat(), "split_sha256": split["split_sha256"],
              "counts_by_source": split.get("counts_by_source"), "candidates": results, "decision": decision,
              "note": "development evidence only (gate cohort); the locked test is run once, for the winner"}
    path = ROOT / "docs/evidence/benchmark" / f"{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=float))
    if mlflow.active_run():
        mlflow.end_run()
    for k, r in results.items():
        s = r.get("score", {})
        print(f"{k:22s} recall {value(s.get('gate', {}).get('defect_recall'))} spec {value(s.get('gate', {}).get('specificity'))} "
              f"low-light {value(s.get('stress', {}).get('low_light', {}).get('defect_recall'))} p95 {s.get('cpu_p95_ms')}")
    print("decision:", json.dumps(decision))
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--adopt"]:
        print(json.dumps(adopt(sys.argv[2]), indent=2))
        raise SystemExit(0)
    raise SystemExit(main(sys.argv[1:] or list(CANDIDATES)))
