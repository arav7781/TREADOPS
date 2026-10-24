"""Damage locator: a YOLO26 detector for local damage (crack, cut, chunk, bulge, foreign object).

The classifier answers "is this tyre defective?" from the whole image and learned mostly aged, cracked rubber; a
fresh cut on a new tyre can slip past it. The locator answers "where is there damage?" and is used only to escalate:
a confident box sends an auto-clear to an expert, it never clears a tyre.

    prepare   public Roboflow Universe exports (CC BY 4.0) -> data/damage/yolo/{images,labels}/{train,val,test}
    train     YOLO26 detection from pinned weights -> artifacts/damage/best.pt (+ MLflow run, experiment damage)
    evaluate  test mAP per class, then the escalation threshold chosen on the policy role's good tyres and its
              effect reported on the calibration role (our own photos, all sources) -> artifacts/damage/
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .common import load_yaml, repo_path, sha256_bytes, sha256_file, write_json

CONFIG = "configs/damage.yaml"
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def _protected_embeddings(stretch: int) -> tuple[np.ndarray, list[dict]]:
    """Reference embeddings of every classification image in a protected role (calibration, policy, test),
    computed on a square stretch like Roboflow's export, so re-annotated copies of our photos are recognised."""
    from PIL import Image

    from .descriptors import ReferenceEncoder
    from .evaluate import read_jsonl
    from .preprocess import PreprocessContract, tensor_from_canonical
    from .weights import pretrained

    rows = [r for role in ("calibration", "policy", "test") for r in read_jsonl(repo_path("data/eval") / f"{role}.jsonl")]
    enc = ReferenceEncoder(str(pretrained("yolo26n-cls.pt")), device="cpu")
    contract = PreprocessContract()
    embs = []
    for i in range(0, len(rows), 64):
        xs = []
        for r in rows[i : i + 64]:
            img = Image.open(repo_path(r["path"])).convert("RGB").resize((stretch, stretch))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=95)
            xs.append(tensor_from_canonical(buf.getvalue(), contract))
        embs.append(enc.embed(np.concatenate(xs)))
    return (np.concatenate(embs) if embs else np.zeros((0, enc.dim), np.float32)), rows


def prepare(config: str = CONFIG) -> dict[str, Any]:
    """Pool every export (their own splits are unusable: one puts all damage in train), drop exact duplicates and
    any image that is a copy of one of OUR protected-role photos, then split 70/15/15 by a hash of the bytes,
    stratified by the image's main damage class."""
    from .descriptors import ReferenceEncoder
    from .preprocess import PreprocessContract, tensor_from_canonical
    from .weights import pretrained

    cfg = load_yaml(config)
    classes = cfg["classes"]
    root = repo_path(cfg["out_dir"]) / "yolo"
    if root.exists():
        shutil.rmtree(root)
    pool: dict[str, dict] = {}
    dropped = Counter()
    for src in cfg["sources"]:
        archive = repo_path(src["archive"])
        if sha256_file(archive) != src["archive_sha256"]:
            raise SystemExit(f"{src['id']}: archive does not match its pinned SHA-256")
        zf = zipfile.ZipFile(archive)
        names = load_yaml_bytes(zf.read("data.yaml"))["names"]
        remap = {i: classes.index(src["class_map"][n]) for i, n in enumerate(names)}
        members = set(zf.namelist())
        for member in sorted(n for n in members if "/images/" in n and not n.endswith("/")):
            data = zf.read(member)
            sha = sha256_bytes(data)
            if sha in pool:
                dropped["exact_duplicate"] += 1
                continue
            label_member = member.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            boxes = []
            if label_member in members:
                for line in zf.read(label_member).decode().splitlines():
                    parts = line.split()
                    if len(parts) == 5:  # boxes only
                        boxes.append([remap[int(parts[0])], *parts[1:]])
            pool[sha] = {"source": src["id"], "member": member, "data": data, "boxes": boxes}

    # Leakage guard: never train or test the locator on a copy of a photo we evaluate the classifier on.
    shas = sorted(pool)
    enc = ReferenceEncoder(str(pretrained("yolo26n-cls.pt")), device="cpu")
    contract = PreprocessContract()
    emb = np.concatenate([enc.embed(np.concatenate([tensor_from_canonical(pool[h]["data"], contract)
                                                    for h in shas[i : i + 64]])) for i in range(0, len(shas), 64)])
    prot, prot_rows = _protected_embeddings(cfg["recipe"]["imgsz"])
    if len(prot):
        sims = emb @ prot.T
        hit = sims.max(1) >= cfg.get("leakage_cosine_min", 0.90)
        for h, is_hit, j in zip(shas, hit, sims.argmax(1), strict=True):
            if is_hit:
                pool[h]["leak"] = prot_rows[int(j)]["image_id"]
    leaks = [h for h in shas if "leak" in pool[h]]
    for h in leaks:
        dropped["copy_of_protected_role_image"] += 1
        del pool[h]

    def main_class(rec) -> str:
        return Counter(classes[b[0]] for b in rec["boxes"]).most_common(1)[0][0] if rec["boxes"] else "background"

    by_cls: dict[str, list[str]] = {}
    for h in sorted(pool):
        by_cls.setdefault(main_class(pool[h]), []).append(h)
    counts: dict[str, Counter] = {s: Counter() for s in ("train", "val", "test")}
    images = Counter()
    for _cls, hs in sorted(by_cls.items()):
        hs = sorted(hs, key=lambda h: sha256_bytes(f"{cfg['recipe']['seed']}:{h}".encode()))
        n_test, n_val = round(len(hs) * 0.15), round(len(hs) * 0.15)
        for k, h in enumerate(hs):
            split = "test" if k < n_test else "val" if k < n_test + n_val else "train"
            rec = pool[h]
            stem = f"{rec['source'].split(':')[0].lower()}-{h[:16]}"
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
            (root / "images" / split / f"{stem}.jpg").write_bytes(rec["data"])
            lines = [" ".join(str(x) for x in b) for b in rec["boxes"]]
            (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            images[split] += 1
            for b in rec["boxes"]:
                counts[split][classes[b[0]]] += 1
    dataset = {"path": str(root), "train": "images/train", "val": "images/val", "test": "images/test",
               "names": {i: c for i, c in enumerate(classes)}}
    (root / "dataset.yaml").write_text(json.dumps(dataset, indent=2))
    manifest = {"version": cfg["version"], "classes": classes, "images": dict(images), "dropped": dict(dropped),
                "images_by_main_class": {c: len(v) for c, v in sorted(by_cls.items())},
                "boxes_by_split": {s: dict(c) for s, c in counts.items()},
                "leakage_guard": {"protected_images_checked": len(prot_rows), "rule": "reference-embedding cosine "
                                  ">= 0.90 to any calibration/policy/test photo (square-stretched like the export)",
                                  "dropped": len(leaks)},
                "sources": [{k: s[k] for k in ("id", "url", "license", "attribution", "archive_sha256")}
                            for s in cfg["sources"]]}
    write_json(repo_path(cfg["out_dir"]) / "manifest.json", manifest)
    return manifest


def load_yaml_bytes(data: bytes) -> dict:
    import yaml

    return yaml.safe_load(data)


def train(config: str = CONFIG, device: str | None = None) -> dict[str, Any]:
    from ultralytics import YOLO, settings

    from .train import _mlflow_run, resolve_device
    from .weights import pinned_sha, pretrained

    cfg = load_yaml(config)
    rec = cfg["recipe"]
    out = repo_path("artifacts/damage")
    out.mkdir(parents=True, exist_ok=True)
    settings.update({"mlflow": False})
    dev = resolve_device(device or "auto")
    mlrun = _mlflow_run("treadops-damage-locator")
    if mlrun:
        import mlflow

        mlflow.log_params({f"recipe.{k}": v for k, v in rec.items()})
        mlflow.set_tags({"purpose": "damage_locator", "data_manifest_sha": sha256_file(
            repo_path(cfg["out_dir"]) / "manifest.json"), "device": dev})
    model = YOLO(str(pretrained(rec["model"])))
    t0 = time.time()
    model.train(data=str(repo_path(cfg["out_dir"]) / "yolo/dataset.yaml"), imgsz=rec["imgsz"], epochs=rec["epochs"],
                batch=rec["batch"], patience=rec["patience"], seed=rec["seed"],
                deterministic=bool(rec.get("deterministic", True)), device=dev,
                project=str(repo_path("runs/damage")), name="locator", exist_ok=True, plots=False, verbose=False)
    best = Path(model.trainer.best)
    shutil.copy2(best, out / "best.pt")
    record = {"recipe": rec, "device": dev, "duration_s": round(time.time() - t0, 1),
              "pretrained": {"name": rec["model"], "sha256": pinned_sha(rec["model"])},
              "best_pt_sha256": sha256_file(out / "best.pt"), "classes": cfg["classes"],
              "mlflow_run_id": mlrun.info.run_id if mlrun else None}
    write_json(out / "training_record.json", record)
    if mlrun:
        import mlflow

        mlflow.log_artifact(str(out / "best.pt"), artifact_path="weights")
        mlflow.end_run()
    return record


def max_confidence(model, rgb_batch: list[np.ndarray], imgsz: int) -> list[tuple[float, str | None]]:
    """Highest box confidence per image (0.0 and None when nothing is found), with that box's class."""
    out = []
    for res in model.predict(source=rgb_batch, imgsz=imgsz, conf=0.05, verbose=False, device="cpu"):
        if len(res.boxes):
            i = int(res.boxes.conf.argmax())
            out.append((float(res.boxes.conf[i]), res.names[int(res.boxes.cls[i])]))
        else:
            out.append((0.0, None))
    return out


def evaluate(config: str = CONFIG) -> dict[str, Any]:
    from ultralytics import YOLO

    from .evaluate import read_jsonl
    from .preprocess import load_canonical

    cfg = load_yaml(config)
    out = repo_path("artifacts/damage")
    model = YOLO(str(out / "best.pt"))
    val = model.val(data=str(repo_path(cfg["out_dir"]) / "yolo/dataset.yaml"), split="test", imgsz=cfg["recipe"]["imgsz"],
                    device="cpu", plots=False, verbose=False)
    per_class = {model.names[int(c)]: round(float(val.box.ap50[i]), 4) for i, c in enumerate(val.box.ap_class_index)}
    detection = {"test_map50": round(float(val.box.map50), 4), "test_map50_95": round(float(val.box.map), 4),
                 "precision": round(float(val.box.mp), 4), "recall": round(float(val.box.mr), 4),
                 "ap50_by_class": per_class}

    # How the locator behaves on OUR photos (every source): threshold on the policy role's good tyres only.
    def scan(role):
        rows = read_jsonl(repo_path("data/eval") / f"{role}.jsonl")
        confs: list[tuple[float, str | None]] = []
        for i in range(0, len(rows), 16):
            batch = [load_canonical(repo_path(r["path"]).read_bytes()) for r in rows[i : i + 16]]
            confs += max_confidence(model, batch, cfg["recipe"]["imgsz"])
        return rows, confs

    pol_rows, pol = scan("policy")
    good = sorted(c for (c, _), r in zip(pol, pol_rows, strict=True) if r["label"] == "good")
    limit = cfg["escalation"]["max_false_escalation"]
    threshold = float(np.quantile(good, 1 - limit)) if good else 1.0
    threshold = min(max(threshold, 0.25), 0.95)  # never below a meaningful box confidence

    def effect(rows, confs):
        by: dict[str, dict] = {}
        for (c, _cls), r in zip(confs, rows, strict=True):
            s = by.setdefault(r.get("source_id", "D1:v1"), {"good": 0, "good_escalated": 0, "defective": 0,
                                                             "defective_flagged": 0})
            if r["label"] == "good":
                s["good"] += 1
                s["good_escalated"] += c >= threshold
            else:
                s["defective"] += 1
                s["defective_flagged"] += c >= threshold
        return {k: {**v, "false_escalation": round(v["good_escalated"] / max(v["good"], 1), 4),
                    "defect_flag_rate": round(v["defective_flagged"] / max(v["defective"], 1), 4)}
                for k, v in sorted(by.items())}

    cal_rows, cal = scan("calibration")
    test_rows, test = scan("test")  # reported only: the threshold above never sees it
    report = {"detection": detection, "escalation_threshold": round(threshold, 4),
              "threshold_rule": f"(1 - {limit}) quantile of max box confidence on policy-role good tyres, "
                                "clamped to [0.25, 0.95]",
              "policy_role": effect(pol_rows, pol), "calibration_role": effect(cal_rows, cal),
              "test_role": effect(test_rows, test),
              "gates": {"test_map50": {"value": detection["test_map50"], "min": cfg["gates"]["test_map50_min"],
                                       "status": "passed" if detection["test_map50"] >= cfg["gates"]["test_map50_min"]
                                       else "failed"}}}
    write_json(out / "evaluation.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["prepare", "train", "evaluate"])
    ap.add_argument("--config", default=CONFIG)
    args = ap.parse_args(argv)
    fn = {"prepare": prepare, "train": train, "evaluate": evaluate}[args.step]
    print(json.dumps(fn(args.config), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
