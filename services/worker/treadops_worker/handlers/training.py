"""Training job: snapshot -> materialised dataset -> train -> evaluate (vs champion) -> parity -> bundle -> register.

Runs on the `training` queue (concurrency 1) and additionally holds the `training-slot` lease, so two fits can
never overlap even with a misconfigured worker. Cancellation is cooperative between epochs.
"""

from __future__ import annotations

import io
import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from treadops_api import audit, jobs
from treadops_api.db import must_get, new_session, session_scope
from treadops_api.models import DatasetSnapshot, ImageAsset, Job, TrainingRun
from treadops_api.services.inspections import active_deployment
from treadops_api.services.releases import import_bundle
from treadops_api.settings import get_settings
from treadops_api.storage import get_storage
from treadops_ml.common import repo_path, sha256_json, write_json

from ..runtime import WORKER_ID, bundles
from ..tasks import JobCancelled, JobContext, Permanent, Transient, handler, on_fail  # noqa: F401

LEASE = "training-slot"


def _materialise(snapshot: DatasetSnapshot, work: Path, db) -> dict:
    storage = get_storage()
    df = pd.read_parquet(io.BytesIO(storage.get(snapshot.artifact_uri.removeprefix(f"s3://{storage.bucket}/"))))
    data = work / "data"
    for r in df.itertuples():
        sub = "train" if r.role == "dev_train" else "val"
        p = data / sub / r.label / f"{r.image_id}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(storage.get(r.canonical_key))
    eval_dir = work / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    protected = db.execute(select(ImageAsset).where(ImageAsset.image_origin == "public_dataset",
                                                    ImageAsset.role.in_(("calibration", "policy", "test")))
                           .order_by(ImageAsset.id)).scalars().all()
    by_role: dict[str, list] = {}
    for a in protected:
        p = work / "eval_images" / f"{a.id}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(storage.get(a.canonical_key))
        by_role.setdefault(a.role, []).append({"image_id": str(a.id), "label": a.label, "group_id": a.group_id,
                                               "role": a.role, "original_sha256": a.original_sha256,
                                               "path": str(p)})
    for role, rows in by_role.items():
        (eval_dir / f"{role}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    # Leakage check for this snapshot against the protected roles (same contract as the DVC split stage).
    snap_groups = set(df.group_id)
    prot_groups = {a.group_id for a in protected}
    snap_hashes = set(df.original_sha256)
    prot_hashes = {a.original_sha256 for a in protected}
    leakage = {"groups_in_multiple_roles": len(snap_groups & prot_groups),
               "original_sha_in_multiple_roles": len(snap_hashes & prot_hashes), "image_ids_unassigned": 0}
    split_manifest = {"snapshot_id": str(snapshot.id), "manifest_sha256": snapshot.manifest_sha256,
                      "leakage": leakage, "split_sha256": sha256_json({"snapshot": snapshot.manifest_sha256,
                                                                      "protected": sorted(prot_hashes)})}
    write_json(work / "split_manifest.json", split_manifest)
    if any(leakage.values()):
        raise Permanent("snapshot_leakage", f"snapshot overlaps protected roles: {leakage}")
    return {"train_images": int((df.role == "dev_train").sum()), "val_images": int((df.role == "dev_val").sum()),
            "protected": {k: len(v) for k, v in by_role.items()}, "leakage": leakage}


@handler("training")
def run(ctx: JobContext) -> dict:
    from treadops_ml import evaluate, export_parity, package, train

    run_id = uuid.UUID(ctx.payload["training_run_id"])
    db = new_session()
    if not jobs.acquire_lease(db, LEASE, f"{WORKER_ID}:{ctx.job_id}", ttl_s=6 * 3600):
        db.close()
        raise Transient("training_slot_busy", "another training run holds the training slot")
    work = repo_path("runs/jobs") / str(ctx.job_id)
    try:
        tr = must_get(db, TrainingRun, run_id)
        snap = must_get(db, DatasetSnapshot, tr.snapshot_id)
        tr.status, tr.started_at = "running", datetime.now(UTC)
        db.commit()
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        ctx.heartbeat(0.02, "materialising snapshot")
        mat = _materialise(snap, work, db)

        def on_epoch(epoch: int, total: int, metrics: dict) -> bool:
            s = new_session()
            try:
                row = must_get(s, TrainingRun, run_id)
                row.epoch_metrics = [*row.epoch_metrics, {"epoch": epoch, **metrics}]
                s.commit()
            finally:
                s.close()
            try:
                ctx.heartbeat(0.05 + 0.75 * epoch / max(total, 1), f"epoch {epoch}/{total}")
            except JobCancelled:
                return True
            return False

        ctx.heartbeat(0.05, "training")
        record = train.run(recipe=tr.config_id, data_root=str(work / "data"), out_dir=str(work / "native"),
                           name=f"job-{str(ctx.job_id)[:8]}", device=get_settings().train_device, on_epoch=on_epoch,
                           extra_tags={"training_run_id": str(run_id), "snapshot_id": str(snap.id)})
        tr.mlflow_run_id = record.get("mlflow_run_id")
        tr.checkpoint_sha256 = record["best_pt_sha256"]
        db.commit()
        if record["status"] == "cancelled":
            raise JobCancelled()

        ctx.heartbeat(0.82, "evaluating")
        _ptr, dep = active_deployment(db)
        db.commit()
        champion = str(bundles().local_dir(dep.bundle_sha256, dep.model_version.bundle_uri)) if dep else None
        summary = evaluate.run(native_dir=str(work / "native"), out_dir=str(work / "evaluation"),
                               eval_dir=str(work / "eval"), champion_bundle=champion, purpose=f"training_run:{run_id}",
                               metrics_path=str(work / "metrics.json"))
        ctx.heartbeat(0.9, "export + parity")
        parity = export_parity.run(str(work / "native"), str(work / "evaluation"), str(work / "export"),
                                   eval_dir=str(work / "eval"))
        ctx.heartbeat(0.94, "packaging")
        pkg = package.run(native_dir=str(work / "native"), evaluation_dir=str(work / "evaluation"),
                          export_dir=str(work / "export"), out_dir=str(work / "bundle"),
                          dataset_snapshot_id=str(snap.id), eval_dir=str(work / "eval"),
                          split_manifest_path=str(work / "split_manifest.json"),
                          summary_metrics_path=str(work / "metrics.json"))
    finally:
        jobs.release_lease(db, LEASE, f"{WORKER_ID}:{ctx.job_id}")
        db.close()

    with session_scope() as s:
        mv = import_bundle(s, get_storage(), work / "bundle", training_run_id=run_id)
        tr = must_get(s, TrainingRun, run_id)
        tr.status, tr.finished_at = "succeeded", datetime.now(UTC)
        tr.model_version_id = mv.id
        tr.artifact_uri = mv.bundle_uri
        tr.metrics = {"summary": summary, "parity": parity, "materialised": mat, "gate_status": pkg["gate_status"],
                      "registry": pkg["registry"], "duration_s": record["duration_s"],
                      "epochs_completed": record["epochs_completed"]}
        audit.record(s, action="training.completed", resource_type="training_run", resource_id=run_id,
                     correlation_id=ctx.correlation_id,
                     payload={"model_version_id": str(mv.id), "gate_status": mv.gate_status,
                              "bundle_sha256": mv.bundle_sha256})
        result = {"training_run_id": str(run_id), "model_version_id": str(mv.id), "gate_status": mv.gate_status,
                  "bundle_sha256": mv.bundle_sha256}
        jobs.succeed(s, must_get(s, Job, ctx.job_id, populate_existing=True), result)
    shutil.rmtree(work / "data", ignore_errors=True)
    shutil.rmtree(work / "eval_images", ignore_errors=True)
    return result


@on_fail("training")
def _failed(db, job: Job, status: str) -> None:
    tr = db.get(TrainingRun, uuid.UUID(job.payload["training_run_id"]))
    if tr is not None:
        tr.status = status
        tr.finished_at = datetime.now(UTC)
        tr.error = job.error_message if status == "failed" else "cancelled between epochs; partial logs preserved"
