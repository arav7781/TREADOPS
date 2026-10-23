"""Inference job: pinned bundle -> stage trace (each image stored) -> subject gate -> calibrated triage -> one atomic
commit. An image that is not a tyre ends `rejected` with no prediction."""

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

from .. import subject_gate, vision_check
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
        emb = reference_encoder().embed(staged["tensor"])[0]
        subject = subject_gate.assess(emb, bundles().local_dir(tup["bundle_sha256"], tup["bundle_uri"]))
        # The vision model is asked about every auto-clear and every unusual image: is a tyre the subject, and is
        # damage visible? It can send a tyre to an expert or reject a non-tyre; it never clears anything.
        ask = vision_check.enabled() and (pred.triage == "good_candidate" or subject["unusual"])
        opinion = vision_check.second_opinion(staged["canonical_rgb"]) if ask else None
        no_tyre = vision_check.sees_no_tyre(opinion)
        subject |= {"vision_tyre_visible": (opinion or {}).get("tyre_visible"),
                    "vision_subject": (opinion or {}).get("subject")}
        # Rejected only when both agree (unlike any training tyre AND no tyre seen); either alone means the verdict
        # is not trusted, so the image is treated as low quality and can never be auto-cleared.
        rejected = subject["unusual"] and no_tyre
        quality_status, quality_reasons = pred.quality_status, list(pred.quality_reasons)
        if not rejected and (subject["unusual"] or no_tyre):
            quality_status = "low_quality"
            quality_reasons.append("unusual_image" if subject["unusual"] else "vision_sees_no_tyre")
        q = pred.descriptors
        vision_note = (f"; vision model: {'no tyre' if no_tyre else 'tyre'} "
                       f"({subject['vision_subject'] or 'no description'})"
                       if subject["vision_tyre_visible"] is not None else "")
        _put_stage(storage, insp.id, Stage(
            5, "quality", "Image quality screen", None,
            {"descriptors": q, "status": "rejected" if rejected else quality_status,
             "reasons": [*quality_reasons, *(["not_a_tyre"] if rejected else [])],
             "thresholds": bundle.policy.quality, "subject": subject},
            0.0,
            [f"brightness {q['brightness']:.3f} · contrast {q['contrast']:.3f} · blur var {q['blur_variance']:.1f}"
             f" · saturation {q['saturation']:.3f}",
             f"subject check: score {subject['score']:.3f} "
             f"({'unusual' if subject['unusual'] else 'typical'}; unusual above {subject['unusual_above']}){vision_note}",
             "REJECTED: not a tyre photo, no verdict is given" if rejected else
             f"quality {quality_status}" + (f" ({', '.join(quality_reasons)})" if quality_reasons else "")
             + "; low quality can never be auto-cleared"]), db)
        if rejected:
            return _reject(ctx, insp.id, {
                "reason": "not_a_tyre", "subject": subject,
                "message": "This image does not show a tyre, so no damage verdict was produced. Upload a photo of the "
                           "tyre: its tread, its sidewall or the whole tyre.",
                "vision": {k: (opinion or {}).get(k) for k in ("model", "prompt_version", "prompt_sha256", "subject",
                                                               "latency_ms")}})
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

        triage, decision, why = vision_check.decide(pred.triage, opinion)
        if quality_status == "low_quality" and triage == "good_candidate":  # low quality is never auto-cleared
            triage = "manual_review"
            why = "; ".join(filter(None, [why, (
                f"image unusual for this model (subject score {subject['score']:.3f} > {subject['unusual_above']})"
                if subject["unusual"] else "vision model sees no tyre in this photo") + ", so not auto-cleared"]))
        triage_reason = f"{pred.triage_reason}; {why}" if why else pred.triage_reason
        # Stored as a second opinion only when it judged an auto-clear (an unusual-image call is in `subject`).
        second = {**opinion, "decision": decision, "policy_triage": pred.triage} \
            if opinion and decision != "not_needed" else None

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
        base = heat if triage != "good_candidate" else thumb(Image.fromarray(staged["canonical_rgb"]), 720)
        final = annotate(base, triage, pred.p_defective, label)
        log = [f"final decision {triage.upper()} · human clearance required"]
        if second:
            seen = second.get("findings") or []
            log.insert(0, f"vision second opinion ({second['model']}, prompt {second['prompt_version']}): " + (
                f"no usable answer ({second.get('error', '')[:60]})" if decision == "unavailable" else
                f"damage {'visible' if second['damage_visible'] else 'not visible'} ({second.get('confidence')})"
                + (f": {', '.join(seen)}" if seen else "")) + f" -> {decision}")
        _put_stage(storage, insp.id, Stage(
            8, "classified", "Classified image", final,
            {"triage": triage, "p_defective": pred.p_defective, "policy_triage": pred.triage,
             **({"second_opinion": second} if second else {})}, 0.0, log), db)

        emb_key = f"embeddings/{insp.id}.npy"
        buf = io.BytesIO()
        np.save(buf, emb)
        storage.put(emb_key, buf.getvalue())
        ctx.heartbeat(0.9, "persisting")
    finally:
        db.close()

    total_ms = (time.perf_counter() - t_start) * 1000
    lineage = {**bundle.lineage(), "image_sha256": asset.original_sha256, "image_id": str(asset.id),
               "deployment_id": tup["deployment_id"], "deployment_generation": tup["generation"],
               "model_version": tup.get("registry_version"), "worker": ctx.owner,
               "policy": {"version": bundle.policy.version, "binary_threshold": bundle.policy.binary_threshold,
                          "triage_low": bundle.policy.triage_low, "triage_high": bundle.policy.triage_high},
               "subject_gate": {k: subject[k] for k in ("version", "method", "unusual_above")},
               **({"second_opinion": {k: second[k] for k in ("model", "prompt_version", "prompt_sha256")}}
                  if second else {})}
    with session_scope() as db:
        job = must_get(db, Job, ctx.job_id, populate_existing=True)
        if job.status != "running":
            raise Permanent("job_state_changed", f"job is {job.status}")
        exists = db.execute(select(Prediction.id).where(Prediction.inspection_id == insp.id)).first()
        if exists is None:  # unique(inspection_id) also guards this
            db.add(Prediction(
                inspection_id=insp.id, raw_probs=pred.raw_probs, calibrated_probs=pred.calibrated_probs,
                p_defective=pred.p_defective, triage=triage, triage_reason=triage_reason, second_opinion=second,
                binary_defective=pred.binary_defective, quality_status=quality_status,
                quality={"descriptors": pred.descriptors, "reasons": quality_reasons, "subject": subject},
                timing={"queue_ms": round(queue_ms, 1), "preprocess_ms": round(preprocess_ms, 2),
                        "inference_ms": round(pred.timing_ms["inference_ms"], 2),
                        "attribution_ms": round(attribution_ms, 1), "worker_total_ms": round(total_ms, 1)},
                lineage=lineage, attribution=attribution, embedding_key=emb_key,
                circularity=recommendation(triage, quality_status),
            ))
        row = must_get(db, Inspection, insp.id, populate_existing=True)
        row.status = "succeeded"
        row.completed_at = datetime.now(UTC)
        result = {"triage": triage, "p_defective": pred.p_defective, "bundle_sha256": bundle.sha,
                  "second_opinion": decision, "subject_score": subject["score"]}
        jobs.succeed(db, job, result)
    return result


def _reject(ctx: JobContext, inspection_id, rejection: dict[str, Any]) -> dict[str, Any]:
    """Not a tyre: the job succeeds (the system worked) and the inspection ends `rejected`, with no prediction."""
    with session_scope() as db:
        job = must_get(db, Job, ctx.job_id, populate_existing=True)
        if job.status != "running":
            raise Permanent("job_state_changed", f"job is {job.status}")
        row = must_get(db, Inspection, inspection_id, populate_existing=True)
        row.status = "rejected"
        row.rejection = rejection
        row.completed_at = datetime.now(UTC)
        result = {"rejected": rejection["reason"], "subject_score": rejection["subject"]["score"]}
        jobs.succeed(db, job, result)
    return result


@on_fail("inference")
def _failed(db: Session, job: Job, status: str) -> None:
    insp = db.get(Inspection, job.payload["inspection_id"])
    if insp is not None and insp.status not in ("succeeded", "rejected"):
        insp.status = status
        insp.completed_at = datetime.now(UTC)
