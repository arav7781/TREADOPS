"""Stage `package`: assemble, hash and register the immutable model bundle.

bundle/
  best.pt  model.onnx (only when parity passed)  class_mapping.json  preprocess.json  calibration.json
  policy.json  metrics.json  gate_report.json  model_card.md  reference_features.parquet  reference_pca.npz
  golden/<image_id>.jpg  golden.json  manifest.json (file hashes + lineage; not itself hashed)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import gates
from .bundle import bundle_sha_from_files
from .common import load_yaml, lock_hash, read_json, repo_path, sha256_file, sha256_json, write_json
from .descriptors import DESCRIPTOR_NAMES, DESCRIPTOR_VERSION, ReferenceEncoder, image_descriptors
from .evaluate import NativeRunner, read_jsonl
from .preprocess import PreprocessContract, load_canonical, tensor_from_canonical
from .weights import pinned_sha, pretrained


def reference_features(rows: list[dict], contract: PreprocessContract, n_components: int, out: Path) -> dict:
    enc = ReferenceEncoder(str(pretrained("yolo26n-cls.pt")), device="cpu")
    descs, embs = [], []
    for i in range(0, len(rows), 64):
        chunk = rows[i : i + 64]
        datas = [repo_path(r["path"]).read_bytes() for r in chunk]
        embs.append(enc.embed(np.concatenate([tensor_from_canonical(d, contract) for d in datas])))
        descs.extend(image_descriptors(load_canonical(d)) for d in datas)
    emb = np.concatenate(embs)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, svd_solver="full", random_state=0).fit(emb)
    proj = pca.transform(emb)
    df = pd.DataFrame({"image_id": [r["image_id"] for r in rows], "group_id": [r["group_id"] for r in rows],
                       "role": [r["role"] for r in rows], "label": [r["label"] for r in rows]})
    for k in DESCRIPTOR_NAMES:
        df[k] = [d[k] for d in descs]
    for j in range(n_components):
        df[f"pc_{j}"] = proj[:, j].astype(np.float32)
    df = df.sort_values("image_id").reset_index(drop=True)
    df.to_parquet(out / "reference_features.parquet", index=False)
    np.savez(out / "reference_pca.npz", mean=pca.mean_.astype(np.float32),
             components=pca.components_.astype(np.float32),
             explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32))
    return {
        "encoder": enc.identity(),
        "pca_components": n_components,
        "pca_sha256": sha256_file(out / "reference_pca.npz"),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "descriptor_version": DESCRIPTOR_VERSION,
        "descriptors": list(DESCRIPTOR_NAMES),
        "cohort": "outer validation (calibration + policy roles)",
        "cohort_ids_sha256": sha256_json(sorted(df.image_id)),
        "n": int(len(df)),
        "distinct_parents": int(df.group_id.nunique()),
    }


def model_card(manifest: dict, gate: dict, test: dict, stress: dict, latency: dict, record: dict) -> str:
    def pct(m):
        if not m or m.get("value") is None:
            return "n/a"
        ci = f", 95% CI {m['ci95'][0]:.3f}-{m['ci95'][1]:.3f}" if m.get("ci95") else ""
        return f"{m['value']:.3f} ({m['numerator']}/{m['denominator']}{ci})"

    tb = test.get("binary", {})
    gb = {t["id"]: t for t in gate["quality_targets"]}
    lines = [
        f"# Model card - {manifest['model_family']} tyre casing triage",
        "",
        f"Weights `{manifest['lineage']['best_pt_sha256'][:16]}` · task `classify` · preprocess `{manifest['serving']['preprocess_version']}`"
        f" · policy `{manifest['serving']['policy_version']}` · gate status **{gate['status'].upper()}**",
        "",
        "## Intended use",
        "Image-level triage of tyre photographs into retread-candidate / expert-review / defect-flagged queues at"
        " a casing intake. A human inspector gives final clearance; the model never certifies a tyre as safe.",
        "",
        "## Data",
        "Mendeley D1 (doi:10.17632/bn7ch8tvyp.1, CC BY 4.0): 1828 unique public photographs after removing 28"
        " byte-identical duplicates, grouped into reviewed same-tyre clusters and split 70/15/15 by group."
        " No Michelin production data is used; operational metadata in replays is simulated.",
        f"Fitted on {record['fitted_image_count']} images (dev_train); checkpoint selected on dev_val.",
        "",
        "## Evidence (actual measurements)",
        f"- Gate cohort (development evidence, outer validation): recall {pct(gb['defect_recall'] | {'denominator': gb['defect_recall']['support']})};"
        f" specificity {pct(gb['specificity'] | {'denominator': gb['specificity']['support']})}",
        f"- Locked test at threshold {tb.get('threshold', 'n/a')}: recall {pct(tb.get('defect_recall'))};"
        f" specificity {pct(tb.get('specificity'))}",
        f"- Test AP {test.get('probabilistic', {}).get('average_precision', 'n/a')}, Brier"
        f" {test.get('probabilistic', {}).get('brier', 'n/a')}",
        "- Stress (paired test parents): " + "; ".join(
            f"{k}: recall {v['binary']['defect_recall'].get('value')}, specificity {v['binary']['specificity'].get('value')}"
            for k, v in stress.get("scenarios", {}).items()),
        f"- CPU model-only latency p95 {latency['model_only'].get('p95_ms', 0):.1f} ms (batch 1, warm,"
        f" {latency['hardware'].get('cpu_brand', latency['hardware'].get('machine'))})",
        "",
        "## Limitations",
        "- Public photographs of mixed passenger/commercial tyres; not shearography or plant imagery.",
        "- Same-tyre clusters were inferred from visual similarity and capture order; residual leakage is possible.",
        "- Image-level labels only: the attribution heatmap is not a defect localisation.",
        "- Sustainability figures shown in the app are estimates from Michelin's published truck/bus factors.",
    ]
    return "\n".join(lines) + "\n"


def register_mlflow(bundle_dir: Path, run_id: str | None, registry_name: str, gate: dict, metrics: dict) -> dict:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri or not run_id:
        return {"status": "skipped", "reason": "no tracking server" if not uri else "training run not tracked"}
    import mlflow
    from mlflow import MlflowClient

    from .mlflow_model import TreadOpsBundleModel

    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})
        mlflow.log_artifact(str(bundle_dir / "gate_report.json"), artifact_path="gates")
        mlflow.log_artifact(str(bundle_dir / "model_card.md"), artifact_path="gates")
        info = mlflow.pyfunc.log_model(
            name="treadops_bundle",
            python_model=TreadOpsBundleModel(),
            artifacts={"bundle": str(bundle_dir)},
            code_paths=[str(repo_path("packages/treadops_ml/treadops_ml"))],
            pip_requirements=["ultralytics==8.4.183", "torch==2.14.1", "torchvision==0.29.1", "numpy", "pandas",
                              "pillow", "opencv-python", "scipy", "pyyaml"],
            registered_model_name=registry_name,
        )
    version = str(info.registered_model_version)
    sha = json.loads((bundle_dir / "manifest.json").read_text())["bundle_sha256"]
    client.set_model_version_tag(registry_name, version, "validation_status", gate["status"])
    client.set_model_version_tag(registry_name, version, "gate_report_uri", f"runs:/{run_id}/gates/gate_report.json")
    client.set_model_version_tag(registry_name, version, "bundle_sha256", sha)
    client.set_registered_model_alias(registry_name, "candidate", version)
    if gate["status"] == gates.PASSED:
        client.set_registered_model_alias(registry_name, "challenger", version)
    return {"status": "registered", "name": registry_name, "version": version, "model_uri": info.model_uri}


def run(config_path: str = "configs/bundle.yaml", native_dir: str | None = None, evaluation_dir: str | None = None,
        export_dir: str = "artifacts/export", out_dir: str | None = None, register: bool = True,
        dataset_snapshot_id: str | None = None, eval_dir: str = "data/eval",
        split_manifest_path: str = "data/manifests/split_manifest.json",
        summary_metrics_path: str = "reports/metrics.json") -> dict[str, Any]:
    cfg = load_yaml(config_path)
    pcfg = load_yaml("configs/policy.yaml")
    native = repo_path(native_dir or cfg["native_dir"])
    evaluation = repo_path(evaluation_dir or cfg["evaluation_dir"])
    export = repo_path(export_dir)
    out = repo_path(out_dir or cfg["out_dir"])
    if out.exists():
        shutil.rmtree(out)
    (out / "golden").mkdir(parents=True)

    record = read_json(native / "training_record.json")
    contract = PreprocessContract.from_dict(record["preprocess"])
    runner = NativeRunner(native / "best.pt", "cpu", contract.imgsz)
    shutil.copy2(native / "best.pt", out / "best.pt")
    parity = read_json(export / "parity.json") if (export / "parity.json").exists() else None
    if parity and parity["status"] == "passed":
        shutil.copy2(export / "model.onnx", out / "model.onnx")

    mapping = runner.mapping
    write_json(out / "class_mapping.json", {"mapping": mapping, "names": {str(k): v for k, v in runner.names.items()},
                                            "positive_label": "defective"})
    write_json(out / "preprocess.json", contract.to_dict())
    shutil.copy2(evaluation / "calibration.json", out / "calibration.json")
    shutil.copy2(evaluation / "policy.json", out / "policy.json")

    roles = {r: read_jsonl(repo_path(eval_dir) / f"{r}.jsonl") for r in ("calibration", "policy", "test")}
    monitoring = reference_features(roles["calibration"] + roles["policy"], contract,
                                    int(cfg["reference_pca_components"]), out)

    # golden warm-up batch: deterministic, balanced, from the policy role
    golden = []
    for label in ("defective", "good"):
        for r in sorted((x for x in roles["policy"] if x["label"] == label), key=lambda x: x["image_id"])[
            : int(cfg["golden_images"]) // 2]:
            shutil.copy2(repo_path(r["path"]), out / "golden" / f"{r['image_id']}.jpg")
            golden.append(r)
    from .calibrate import TemperatureCalibrator
    from .policy import Policy

    cal = TemperatureCalibrator.from_dict(read_json(evaluation / "calibration.json"))
    pol = Policy.from_dict(read_json(evaluation / "policy.json"))
    gx = np.concatenate([tensor_from_canonical((out / "golden" / f"{r['image_id']}.jpg").read_bytes(), contract)
                         for r in golden])
    graw = runner.raw(gx)
    write_json(out / "golden.json", {"tolerance_abs": 1e-4, "items": [
        {"image_id": r["image_id"], "label": r["label"], "file": f"golden/{r['image_id']}.jpg",
         "raw_probs": [float(v) for v in graw[i]],
         "p_defective": float(cal.apply(graw[i])[mapping["defective"]]),
         "triage": pol.decide(float(cal.apply(graw[i])[mapping["defective"]]))[0]} for i, r in enumerate(golden)]})

    split_manifest = read_json(split_manifest_path)
    gate_metrics = read_json(evaluation / "gate_metrics.json")
    test = read_json(evaluation / "test_metrics.json")
    stress = read_json(evaluation / "stress.json")
    latency = read_json(evaluation / "latency.json")
    comparative = read_json(evaluation / "comparative.json")
    lineage = {
        "git_sha": record.get("git_sha"),
        "git_dirty": record.get("git_dirty"),
        "split_sha256": record.get("split_sha256"),
        "recipe_sha256": record.get("recipe_sha256"),
        "uv_lock_sha256": record.get("uv_lock_sha256") or lock_hash(),
        "pretrained_sha256": pinned_sha(record["pretrained"]["name"]),
        "best_pt_sha256": record["best_pt_sha256"],
        "preprocess_sha256": contract.sha256,
        "mlflow_run_id": record.get("mlflow_run_id"),
        "training_run_name": record["run_name"],
        "dataset_snapshot_id": dataset_snapshot_id,
    }
    gate = gates.build(
        split_manifest=split_manifest, training_record=record,
        eval_role_ids={r: {x["image_id"] for x in roles[r]} for r in roles},
        gate_metrics=gate_metrics, parity=parity, comparative=comparative, class_mapping=mapping,
        lineage={k: v for k, v in lineage.items() if k not in ("git_dirty", "dataset_snapshot_id", "training_run_name")},
        policy_cfg=pcfg,
    )
    write_json(out / "gate_report.json", gate)
    metrics = {
        "gate_cohort": gate_metrics, "locked_test": test, "stress": stress, "latency": latency,
        "comparative": comparative, "parity": parity, "training": {
            "epochs_completed": record["epochs_completed"], "duration_s": record["duration_s"],
            "device": record["device"], "final_best_pt_validation": record.get("final_best_pt_validation"),
            "fitted_image_count": record["fitted_image_count"], "train_pool_image_count": record["train_pool_image_count"]},
    }
    write_json(out / "metrics.json", metrics)

    files_for_hash = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ("manifest.json", "model_card.md"))
    files = {str(p.relative_to(out)): sha256_file(p) for p in files_for_hash}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task": "classify",
        "model_family": Path(record["pretrained"]["name"]).stem,
        "bundle_sha256": None,
        "registry": {"name": cfg["registry_name"], "version": None},
        "reproducibility": {k: lineage[k] for k in ("git_sha", "git_dirty", "split_sha256", "recipe_sha256",
                                                    "uv_lock_sha256", "pretrained_sha256")},
        "training": {"mlflow_run_id": record.get("mlflow_run_id"), "run_name": record["run_name"],
                     "effective_train_ids_sha256": sha256_json(record["effective_train_ids"]),
                     "fitted_image_count": record["fitted_image_count"], "hardware": record["hardware"],
                     "packages": record["packages"]},
        "evaluation": {"role_hashes": {r: sha256_json(sorted(x["image_id"] for x in roles[r])) for r in roles},
                       "role_counts": {r: len(roles[r]) for r in roles}, "gate_status": gate["status"]},
        "serving": {"runtimes": ["native"] + (["onnx"] if (out / "model.onnx").exists() else []),
                    "approved_device": "cpu", "precision": "fp32", "preprocess_version": contract.version,
                    "preprocess_sha256": contract.sha256, "class_names": {str(k): v for k, v in runner.names.items()},
                    "calibration_version": cal.version, "policy_version": pol.version},
        "monitoring": monitoring,
        "lineage": lineage,
    }
    (out / "model_card.md").write_text(model_card(manifest, gate, test, stress, latency, record))
    files["model_card.md"] = sha256_file(out / "model_card.md")
    manifest["files"] = dict(sorted(files.items()))
    manifest["bundle_sha256"] = bundle_sha_from_files(manifest["files"])
    write_json(out / "manifest.json", manifest)

    summary_metrics = read_json(summary_metrics_path) if repo_path(summary_metrics_path).exists() else {}
    reg = register_mlflow(out, record.get("mlflow_run_id"), cfg["registry_name"], gate, summary_metrics) \
        if register else {"status": "skipped"}
    if reg.get("version"):
        manifest["registry"]["version"] = reg["version"]
        write_json(out / "manifest.json", manifest)
    write_json(out.parent / "bundle_registration.json", reg)
    return {"bundle_sha256": manifest["bundle_sha256"], "gate_status": gate["status"], "registry": reg,
            "files": len(manifest["files"])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="assemble + register the model bundle")
    ap.add_argument("--config", default="configs/bundle.yaml")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args(argv)
    print(json.dumps(run(args.config, register=not args.no_register), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
