"""Second model family: torchvision classifiers (ConvNeXt-Tiny), trained and served on the same contract as YOLO.

The input is the identical letterboxed [0, 1] RGB tensor from `preprocess`; ImageNet normalisation is part of the
module, so ONNX export, parity and serving see the same input as the YOLO family. Augmentation is the same
`TrainTransform`. A checkpoint (best.pt) is a plain dict that loads with `weights_only=True`:

    {"family": "torchvision", "arch": "convnext_tiny", "class_names": {0: "defective", 1: "good"}, "state_dict": ...}

YOLO checkpoints need Ultralytics' pickled classes and cannot load that way, which is how `is_checkpoint` tells the
two families apart.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from .preprocess import PreprocessContract, load_canonical

FAMILY = "torchvision"
CLASSES = ["defective", "good"]  # folder order of data/yolo/dev/<split>/<class>, as YOLO reads it
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build(arch: str, num_classes: int = len(CLASSES)):
    import torch
    import torchvision
    from torch import nn

    if arch != "convnext_tiny":
        raise ValueError(f"unsupported torchvision arch {arch!r}")
    net = torchvision.models.convnext_tiny()
    net.classifier[2] = nn.Linear(net.classifier[2].in_features, num_classes)

    class Normalised(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

        def forward(self, x):
            return self.inner((x - self.mean) / self.std)

    return Normalised(net)


def is_checkpoint(path: str | Path) -> bool:
    import torch

    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:  # an Ultralytics checkpoint (pickled classes), or not a checkpoint at all
        return False
    return isinstance(ckpt, dict) and ckpt.get("family") == FAMILY


def load(path: str | Path):
    """-> (eval-mode CPU module, {index: class name})"""
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    names = {int(k): v for k, v in ckpt["class_names"].items()}
    model = build(ckpt["arch"], len(names))
    model.inner.load_state_dict(ckpt["state_dict"])
    return model.eval(), names


def raw_probs(model, x: np.ndarray, batch: int = 32) -> np.ndarray:
    """(N,3,S,S) prepared tensors -> (N,C) softmax probabilities, model-index order (like YOLO's probs)."""
    import torch

    out = []
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            logits = model(torch.from_numpy(np.ascontiguousarray(x[i : i + batch], dtype=np.float32)))
            out.append(torch.softmax(logits, dim=1).numpy().astype(np.float64))
    return np.concatenate(out)


def export_onnx(model, path: str | Path, imgsz: int) -> Path:
    """Softmax is part of the graph, so ONNX outputs probabilities exactly like the native runner."""
    import torch
    from torch import nn

    class WithSoftmax(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return torch.softmax(self.inner(x), dim=1)

    # TorchScript exporter (deprecated, still supported): the dynamo exporter needs onnxscript, not a dependency.
    torch.onnx.export(WithSoftmax(model).eval(), (torch.zeros(1, 3, imgsz, imgsz),), str(path), input_names=["images"],
                      output_names=["probs"], dynamic_axes={"images": {0: "batch"}, "probs": {0: "batch"}},
                      opset_version=17, dynamo=False)
    return Path(path)


class _Folder:
    """(path, class) samples -> (CHW tensor, class); loaded and augmented inside DataLoader workers."""

    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, cls = self.samples[i]
        return self.transform(load_canonical(path.read_bytes())), cls


def _samples(root: Path, split: str) -> list[tuple[Path, int]]:
    return [(p, i) for i, c in enumerate(CLASSES) for p in sorted((root / split / c).glob("*.jpg"))]


EpochHook = Callable[[int, int, dict[str, float]], bool]


def train(rec: dict[str, Any], data: Path, out: Path, device: str, pretrained_path: Path,
          on_epoch: EpochHook | None = None, mlrun=None) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    """Fine-tune on data/<train|val>/<class>; keeps the lowest-val-loss epoch as out/best.pt.
    Returns (epoch metrics, cancelled, effective training arguments)."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from .transforms import AugmentConfig, EvalTransform, TrainTransform

    seed = int(rec["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    contract = PreprocessContract(imgsz=int(rec["imgsz"]))
    train_t, eval_t = TrainTransform(contract, AugmentConfig(**rec.get("augment", {}))), EvalTransform(contract)
    model = build(rec["arch"])
    imagenet = torch.load(str(pretrained_path), map_location="cpu", weights_only=True)
    missing, unexpected = model.inner.load_state_dict(
        {k: v for k, v in imagenet.items() if not k.startswith("classifier.2.")}, strict=False)
    if unexpected or sorted(missing) != ["classifier.2.bias", "classifier.2.weight"]:
        raise RuntimeError(f"pretrained weights do not match {rec['arch']}: missing={missing} unexpected={unexpected}")
    model = model.to(device)  # everything but the new 2-class head starts from ImageNet
    args = {"arch": rec["arch"], "epochs": int(rec["epochs"]), "imgsz": int(rec["imgsz"]), "batch": int(rec["batch"]),
            "optimizer": "AdamW", "lr0": float(rec["lr0"]), "weight_decay": float(rec.get("weight_decay", 0.05)),
            "label_smoothing": float(rec.get("label_smoothing", 0.1)), "patience": int(rec["patience"]),
            "scheduler": "cosine", "workers": int(rec.get("workers", 4)), "device": device, "seed": seed,
            "checkpoint_rule": "lowest val loss"}
    opt = torch.optim.AdamW(model.parameters(), lr=args["lr0"], weight_decay=args["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args["epochs"])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args["label_smoothing"])
    # Workers are seeded from the generator (python `random`, used by TrainTransform, included), so runs repeat.
    loaders: dict[str, Any] = {
        k: DataLoader(cast(Any, _Folder(_samples(data, k), t)), batch_size=args["batch"], shuffle=k == "train",
                      num_workers=args["workers"], persistent_workers=args["workers"] > 0,
                      generator=torch.Generator().manual_seed(seed))
        for k, t in (("train", train_t), ("val", eval_t))}

    best, bad, t0, epochs, cancelled = float("inf"), 0, time.time(), [], False
    for epoch in range(1, args["epochs"] + 1):
        model.train()
        tl, n = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tl, n = tl + loss.item() * len(y), n + len(y)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        model.eval()
        vl, correct, m = 0.0, 0, 0
        with torch.inference_mode():
            for x, y in loaders["val"]:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vl += loss_fn(logits, y).item() * len(y)
                correct += int((logits.argmax(1) == y).sum())
                m += len(y)
        metrics = {"train_loss": tl / n, "val/loss": vl / m, "metrics/accuracy_top1": correct / m, "lr": lr}
        epochs.append({"epoch": epoch, **metrics, "t": round(time.time() - t0, 2)})
        if mlrun:
            import mlflow

            mlflow.log_metrics({k.replace("/", "_"): v for k, v in metrics.items()}, step=epoch)
        if metrics["val/loss"] < best - 1e-4:
            best, bad = metrics["val/loss"], 0
            torch.save({"family": FAMILY, "arch": rec["arch"], "class_names": dict(enumerate(CLASSES)),
                        "state_dict": {k: v.detach().cpu() for k, v in model.inner.state_dict().items()}},
                       out / "best.pt")
        else:
            bad += 1
        if on_epoch and on_epoch(epoch, args["epochs"], metrics):
            cancelled = True
            break
        if bad >= args["patience"]:
            break
    return epochs, cancelled, args
