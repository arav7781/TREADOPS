"""Inference job: pinned bundle -> stage trace (each image stored) -> calibrated triage -> one atomic commit."""

from __future__ import annotations

import hashlib
import io
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
from PIL import Image
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from treadops_api import jobs
from treadops_api.circularity import recommendation
from treadops_api.db import must_get, new_session, session_scope
from treadops_api.models import ImageAsset, Inspection, InspectionStage, Job, Prediction
from treadops_api.storage import get_storage
from treadops_ml.attribution import annotate, occlusion_map, render_heatmap
from treadops_ml.preprocess import Stage, run_stages, thumb

from ..runtime import bundles, reference_encoder
from ..tasks import JobContext, Permanent, handler, on_fail


def _jpeg(img, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _record_stage(db: Session, inspection_id, stage: Stage, key: str | None) -> None:
    """Upsert so a redelivered job rewrites the same rows instead of duplicating them."""
    values = {"inspection_id": inspection_id, "stage_index": stage.index, "name": stage.name, "title": stage.title,
              "object_key": key, "params": stage.params, "log": stage.log, "duration_ms": float(stage.duration_ms)}
    stmt = insert(InspectionStage).values(**values)
    db.execute(stmt.on_conflict_do_update(constraint="uq_stage_index", set_={
        k: stmt.excluded[k] for k in ("name", "title", "object_key", "params", "log", "duration_ms")}))
    db.commit()


def _put_stage(storage, inspection_id, stage: Stage, db: Session) -> None:
    key = None
    if stage.image is not None:
        key = f"inspections/{inspection_id}/stages/{stage.index:02d}_{stage.name}.jpg"
        storage.put(key, _jpeg(stage.image), "image/jpeg")
    _record_stage(db, inspection_id, stage, key)


@handler("inference")
def run(ctx: JobContext) -> dict[str, Any]:
    storage = get_storage()
    t_start = time.perf_counter()
    db = new_session()
    try:
        insp = db.get(Inspection, ctx.payload["inspection_id"])
        if insp is None:
            raise Permanent("inspection_missing", "inspection row not found")
        asset = must_get(db, ImageAsset, insp.image_id)
        tup = dict(insp.model_tuple)
        insp.status = "running"
        db.commit()
        queue_ms = (datetime.now(UTC) - insp.created_at).total_seconds() * 1000

        bundle = bundles().get(tup["bundle_sha256"], tup["bundle_uri"])  # pinned at acceptance, never "latest"
        sanitized = storage.get(asset.object_key)
        info = asset.ingest_info or {}
        t0 = time.perf_counter()
        staged = run_stages(sanitized, info, bundle.preprocess)
        preprocess_ms = (time.perf_counter() - t0) * 1000
        for st in staged["stages"]:
            _put_stage(storage, insp.id, st, db)
        ctx.heartbeat(0.4, "preprocessed")
        if not asset.canonical_key:
            key = f"canonical/{asset.id}.jpg"
            storage.put(key, staged["canonical_bytes"], "image/jpeg")
            asset.canonical_key = key
            asset.canonical_sha256 = hashlib.sha256(staged["canonical_bytes"]).hexdigest()
            db.commit()

        pred = bundle.predict_canonical_rgb(staged["canonical_rgb"], staged["tensor"])
        q = pred.descriptors
        _put_stage(storage, insp.id, Stage(
            5, "quality", "Image quality screen", None,
            {"descriptors": q, "status": pred.quality_status, "reasons": pred.quality_reasons,
             "thresholds": bundle.policy.quality},
            0.0,
            [f"brightness {q['brightness']:.3f} · contrast {q['contrast']:.3f} · blur var {q['blur_variance']:.1f}"
             f" · saturation {q['saturation']:.3f}",
             f"quality {pred.quality_status}" + (f" ({', '.join(pred.quality_reasons)})" if pred.quality_reasons
                                                 else "") + "; low quality can never be auto-cleared"]), db)
        _put_stage(storage, insp.id, Stage(
            6, "inference", f"YOLO inference ({tup.get('model_family')})", None,
            {"raw_probs": pred.raw_probs, "calibrated_probs": pred.calibrated_probs,
             "temperature": bundle.calibrator.temperature, "p_defective": pred.p_defective,
             "binary_threshold": bundle.policy.binary_threshold, "triage": pred.triage,
             "triage_reason": pred.triage_reason, "bundle_sha256": bundle.sha},
            pred.timing_ms["inference_ms"],
            [f"raw softmax defective={pred.raw_probs['defective']:.4f} good={pred.raw_probs['good']:.4f}",
             f"temperature T={bundle.calibrator.temperature:.3f} -> calibrated p(defective)={pred.p_defective:.4f}",
             f"policy {bundle.policy.version}: {pred.triage} ({pred.triage_reason})"]), db)
        ctx.heartbeat(0.6, "classified")

        t1 = time.perf_counter()
        occ = occlusion_map(bundle, staged["tensor"], grid=8)
        heat = render_heatmap(staged["canonical_rgb"], occ["delta"], staged["letterbox_meta"])
        attribution_ms = (time.perf_counter() - t1) * 1000
        top = np.unravel_index(np.argmax(occ["delta"]), occ["delta"].shape)
        attribution = {"method": "occlusion_sensitivity", "signal": occ["signal"], "grid": occ["grid"],
                       "max_delta": float(occ["delta"].max()),
                       "min_delta": float(occ["delta"].min()), "strongest_cell": [int(top[0]), int(top[1])],
                       "note": "attribution heatmap, not a defect segmentation"}
        _put_stage(storage, insp.id, Stage(
            7, "attribution", "Occlusion attribution (8x8)", thumb(heat, 720), attribution, attribution_ms,
            [f"64 occluded copies in one batch; largest drop in the defective-vs-good logit margin "
             f"{occ['delta'].max():+.2f} (base margin {occ['base_margin']:+.2f})",
             "red = regions supporting 'defective', blue = regions supporting 'good'",
             "explanation aid only - not a defect localisation"]), db)

        label = f"{tup.get('model_family')} v{tup.get('registry_version') or '?'}"
        base = heat if pred.triage != "good_candidate" else thumb(Image.fromarray(staged["canonical_rgb"]), 720)
        final = annotate(base, pred.triage, pred.p_defective, label)
        _put_stage(storage, insp.id, Stage(
            8, "classified", "Classified image", final, {"triage": pred.triage, "p_defective": pred.p_defective},
            0.0, [f"final decision {pred.triage.upper()} · human clearance required"]), db)

        emb_key = f"embeddings/{insp.id}.npy"
        buf = io.BytesIO()
        np.save(buf, reference_encoder().embed(staged["tensor"])[0])
        storage.put(emb_key, buf.getvalue())
        ctx.heartbeat(0.9, "persisting")
    finally:
        db.close()

    total_ms = (time.perf_counter() - t_start) * 1000
    lineage = {**bundle.lineage(), "image_sha256": asset.original_sha256, "image_id": str(asset.id),
               "deployment_id": tup["deployment_id"], "deployment_generation": tup["generation"],
               "model_version": tup.get("registry_version"), "worker": ctx.owner}
    with session_scope() as db:
        job = must_get(db, Job, ctx.job_id, populate_existing=True)
        if job.status != "running":
            raise Permanent("job_state_changed", f"job is {job.status}")
        exists = db.execute(select(Prediction.id).where(Prediction.inspection_id == insp.id)).first()
        if exists is None:  # unique(inspection_id) also guards this
            db.add(Prediction(
                inspection_id=insp.id, raw_probs=pred.raw_probs, calibrated_probs=pred.calibrated_probs,
                p_defective=pred.p_defective, triage=pred.triage, triage_reason=pred.triage_reason,
                binary_defective=pred.binary_defective, quality_status=pred.quality_status,
                quality={"descriptors": pred.descriptors, "reasons": pred.quality_reasons},
                timing={"queue_ms": round(queue_ms, 1), "preprocess_ms": round(preprocess_ms, 2),
                        "inference_ms": round(pred.timing_ms["inference_ms"], 2),
                        "attribution_ms": round(attribution_ms, 1), "worker_total_ms": round(total_ms, 1)},
                lineage=lineage, attribution=attribution, embedding_key=emb_key,
                circularity=recommendation(pred.triage, pred.quality_status),
            ))
        row = must_get(db, Inspection, insp.id, populate_existing=True)
        row.status = "succeeded"
        row.completed_at = datetime.now(UTC)
        result = {"triage": pred.triage, "p_defective": pred.p_defective, "bundle_sha256": bundle.sha}
        jobs.succeed(db, job, result)
    return result


@on_fail("inference")
def _failed(db: Session, job: Job, status: str) -> None:
    insp = db.get(Inspection, job.payload["inspection_id"])
    if insp is not None and insp.status not in ("succeeded",):
        insp.status = status
        insp.completed_at = datetime.now(UTC)
