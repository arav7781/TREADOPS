"""Stage `train`: real Ultralytics YOLO26 classification training with the full-image trainer.

Writes <out_dir>/best.pt plus training_record.json (recipe, effective data IDs, seed, lock hash, hardware,
pretrained hash, epoch series) and logs one project-owned MLflow run when a tracking server is reachable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .common import (
    git_state,
    hardware,
    load_yaml,
    lock_hash,
    package_versions,
    read_json,
    repo_path,
    resolve_device,
    sha256_file,
    sha256_json,
    write_json,
)
from .preprocess import PreprocessContract

EpochHook = Callable[[int, int, dict[str, float]], bool]  # (epoch, total, metrics) -> stop requested


class TrainingCancelled(RuntimeError):
    pass


def _folder_ids(root: Path) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for sub in ("train", "val"):
        out[sub] = {}
        for cls_dir in sorted((root / sub).iterdir()) if (root / sub).exists() else []:
            if cls_dir.is_dir():
                out[sub][cls_dir.name] = sorted(p.stem for p in cls_dir.iterdir() if p.suffix == ".jpg")
    return out


def load_recipe(recipe: str) -> dict[str, Any]:
    p = repo_path(recipe if recipe.endswith(".yaml") else f"configs/recipes/{recipe}.yaml")
    return load_yaml(p)


def _mlflow_run(experiment: str):
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        return mlflow.start_run()
    except Exception as exc:  # tracking outage must not lose a training run; it is recorded instead
        print(f"[train] MLflow unavailable ({exc.__class__.__name__}: {exc}); continuing without tracking")
        return None


def run(
    config_path: str = "configs/train.yaml",
    recipe: str | None = None,
    data_root: str | None = None,
    out_dir: str | None = None,
    name: str | None = None,
    device: str | None = None,
    on_epoch: EpochHook | None = None,
    extra_tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLO, settings

    from .transforms import AugmentConfig, FullImageDataset, FullImageTrainer
    from .weights import pinned_sha, pretrained

    cfg = load_yaml(config_path)
    rec = load_recipe(recipe or cfg["recipe"])
    data = repo_path(data_root or cfg["data_root"])
    out = repo_path(out_dir or cfg["out_dir"])
    dev = resolve_device(device or cfg.get("device"))
    run_name = name or f"{rec['id']}_seed{rec['seed']}"

    settings.update({"mlflow": False})  # the project owns the MLflow run (no competing autologging)
    weights = pretrained(rec["model"])
    FullImageDataset.augment_config = AugmentConfig(**rec.get("augment", {}))
    contract = PreprocessContract(imgsz=int(rec["imgsz"]))

    ids = _folder_ids(data)
    epochs: list[dict[str, Any]] = []
    final_eval: dict[str, float] = {}
    mlrun = _mlflow_run(cfg.get("mlflow_experiment", "treadops-yolo-classify"))
    recipe_sha = sha256_json(rec)
    split_manifest = repo_path(cfg.get("split_manifest", "data/manifests/split_manifest.json"))
    split_sha = read_json(split_manifest)["split_sha256"] if split_manifest.exists() else None
    if mlrun:
        import mlflow

        mlflow.log_params({f"recipe.{k}": v for k, v in rec.items() if not isinstance(v, dict)})
        mlflow.log_params({f"augment.{k}": v for k, v in rec.get("augment", {}).items()})
        mlflow.set_tags({"recipe_sha": recipe_sha, "split_sha": split_sha or "", "device": dev,
                         "preprocess_sha": contract.sha256, **(extra_tags or {})})

    model = YOLO(str(weights))

    def _on_fit_epoch_end(trainer):
        m = {k: float(v) for k, v in (trainer.metrics or {}).items() if isinstance(v, int | float)}
        tl = getattr(trainer, "tloss", None)  # running mean of loss items: dict in 8.4.x, tensor in older releases
        if isinstance(tl, dict):
            m.update({f"train_{k}": float(v) for k, v in tl.items()})
        elif tl is not None:
            m["train_loss"] = float(tl.mean())
        m["lr"] = float(trainer.optimizer.param_groups[0]["lr"])
        e = int(trainer.epoch) + 1
        if epochs and epochs[-1]["epoch"] == e:  # Ultralytics re-fires this hook for the final best.pt validation
            final_eval.update(m)
            return
        epochs.append({"epoch": e, **m, "t": round(time.time() - t0, 2)})
        if mlrun:
            import mlflow

            mlflow.log_metrics({k.replace("/", "_").replace("(", "").replace(")", ""): v for k, v in m.items()
                                if v == v}, step=e)
        if on_epoch and on_epoch(e, int(trainer.epochs), m):
            trainer.stop = True
            trainer._treadops_cancelled = True

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    t0 = time.time()
    train_args = dict(
        trainer=FullImageTrainer,
        data=str(data),
        epochs=int(rec["epochs"]),
        imgsz=int(rec["imgsz"]),
        batch=int(rec["batch"]),
        device=dev,
        seed=int(rec["seed"]),
        deterministic=True,
        workers=int(rec.get("workers", 2)),
        cache=False,
        optimizer=rec["optimizer"],
        lr0=float(rec["lr0"]),
        patience=int(rec["patience"]),
        fraction=float(rec.get("fraction", 1.0)),
        project=str(repo_path(cfg.get("runs_dir", "runs/treadops"))),
        name=run_name,
        exist_ok=True,
        plots=False,
        verbose=False,
        val=True,
    )
    model.train(**train_args)
    duration = time.time() - t0
    trainer = model.trainer
    cancelled = bool(getattr(trainer, "_treadops_cancelled", False))

    best = Path(trainer.best)
    if not best.exists():
        raise RuntimeError("training finished without best.pt")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out / "best.pt")
    names = {int(k): v for k, v in YOLO(str(out / "best.pt")).names.items()}
    if sorted(names.values()) != ["defective", "good"]:
        raise RuntimeError(f"unexpected class names {names}")

    fitted = sum(len(v) for v in ids["train"].values())
    record = {
        "schema_version": 1,
        "run_name": run_name,
        "status": "cancelled" if cancelled else "completed",
        "recipe": rec,
        "recipe_sha256": recipe_sha,
        "train_args": {k: (v.__name__ if isinstance(v, type) else v) for k, v in train_args.items()},
        "device": dev,
        "duration_s": round(duration, 1),
        "epochs_completed": len(epochs),
        "epoch_metrics": epochs,
        "final_best_pt_validation": final_eval,
        "class_names": names,
        "defective_index": next(k for k, v in names.items() if v == "defective"),
        "pretrained": {"name": rec["model"], "sha256": pinned_sha(rec["model"])},
        "best_pt_sha256": sha256_file(out / "best.pt"),
        "data_root": str(data.relative_to(repo_path("."))) if data.is_relative_to(repo_path(".")) else str(data),
        "effective_train_ids": ids["train"],
        "effective_val_ids": ids["val"],
        "fitted_image_count": fitted if rec.get("fraction", 1.0) >= 1.0 else round(fitted * rec["fraction"]),
        "train_pool_image_count": fitted + sum(len(v) for v in ids["val"].values()),
        "split_sha256": split_sha,
        "preprocess": contract.to_dict(),
        "preprocess_sha256": contract.sha256,
        "uv_lock_sha256": lock_hash(),
        "packages": package_versions(),
        "hardware": hardware(),
        "nondeterminism_note": "seed + deterministic=True; bitwise GPU/MPS reproducibility is not guaranteed",
        **git_state(),
        "mlflow_run_id": mlrun.info.run_id if mlrun else None,
    }
    write_json(out / "training_record.json", record)
    with (out / "epochs.jsonl").open("w") as fh:
        for e in epochs:
            fh.write(json.dumps(e) + "\n")
    if mlrun:
        import mlflow

        mlflow.log_artifact(str(out / "training_record.json"))
        mlflow.log_artifact(str(out / "best.pt"), artifact_path="weights")
        mlflow.set_tag("status", record["status"])
        mlflow.end_run()
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train YOLO26 classification with the full-image trainer")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--recipe", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    rec = run(args.config, args.recipe, args.data_root, args.out_dir, args.name, args.device)
    summary = {k: rec[k] for k in ("run_name", "status", "device", "duration_s", "epochs_completed",
                                   "fitted_image_count", "best_pt_sha256", "mlflow_run_id")}
    summary["last_epoch"] = rec["epoch_metrics"][-1] if rec["epoch_metrics"] else None
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
