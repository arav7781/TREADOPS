"""InspectionService + ReviewService.

Acceptance order (spec §5): validate + sanitize -> write object (before SQL) -> one transaction with asset,
inspection pinned to the active deployment tuple, job and outbox event -> 202. Completion happens in a worker.
"""

from __future__ import annotations

import json
import uuid
from functools import lru_cache
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from treadops_ml.common import sha256_bytes, sha256_json
from treadops_ml.preprocess import ImageRejected, Limits, PreprocessContract, ingest

from .. import audit, jobs
from ..errors import ApiError
from ..models import (
    Deployment,
    EnvironmentPointer,
    ImageAsset,
    Inspection,
    InspectionStage,
    Job,
    JobEvent,
    Prediction,
    ReviewLabel,
    User,
)
from ..security import Principal, sign_media
from ..settings import get_settings
from ..storage import Storage, StorageUnavailable

ENV = "local"
SOURCES = {"upload", "replay", "copilot", "api", "camera"}
METADATA_ORIGINS = {"simulated", "user_provided", "observed"}
TYRE_CATEGORIES = {"passenger", "light_truck", "truck_bus", "two_wheeler", "unknown"}


def active_deployment(db: Session, lock: bool = False) -> tuple[EnvironmentPointer, Deployment | None]:
    q = select(EnvironmentPointer).where(EnvironmentPointer.env == ENV)
    if lock:
        q = q.with_for_update()
    ptr = db.execute(q).scalar_one_or_none()
    if ptr is None:
        ptr = EnvironmentPointer(env=ENV, generation=0)
        db.add(ptr)
        db.flush()
    dep = db.get(Deployment, ptr.active_deployment_id) if ptr.active_deployment_id else None
    return ptr, dep


def model_tuple(dep: Deployment) -> dict[str, Any]:
    mv = dep.model_version
    return {
        "deployment_id": str(dep.id),
        "generation": dep.generation,
        "model_version_id": str(mv.id),
        "registry_version": mv.registry_version,
        "bundle_sha256": dep.bundle_sha256,
        "bundle_uri": mv.bundle_uri,
        "preprocess_version": dep.preprocess_version,
        "calibration_version": dep.calibration_version,
        "policy_version": dep.policy_version,
        "model_family": mv.model_family,
    }


def upload_limits() -> Limits:
    return _limits()


def _limits() -> Limits:
    s = get_settings()
    return Limits(max_bytes=s.max_upload_bytes, max_decoded_pixels=s.max_decoded_pixels)


def accept(
    db: Session,
    storage: Storage,
    *,
    actor: Principal | None,
    actor_id: uuid.UUID | None,
    raw: bytes,
    idempotency_key: str,
    source: str = "upload",
    metadata_origin: str = "user_provided",
    line_id: str | None = None,
    camera_id: str | None = None,
    tyre_category: str | None = None,
    notes: str | None = None,
    parent_image_id: uuid.UUID | None = None,
    transform: dict[str, Any] | None = None,
    transform_hash: str | None = None,
    scenario_id: str | None = None,
    replay_job_id: uuid.UUID | None = None,
    replay_sequence: int | None = None,
    enforce_backpressure: bool = True,
) -> tuple[Inspection, bool]:
    if not idempotency_key or len(idempotency_key) > 200:
        raise ApiError(400, "missing_idempotency_key", "An Idempotency-Key header (<=200 chars) is required.")
    if source not in SOURCES:
        raise ApiError(422, "invalid_source", f"source must be one of {sorted(SOURCES)}")
    if metadata_origin not in METADATA_ORIGINS:
        raise ApiError(422, "invalid_metadata_origin", f"metadata_origin must be one of {sorted(METADATA_ORIGINS)}")
    if tyre_category and tyre_category not in TYRE_CATEGORIES:
        raise ApiError(422, "invalid_tyre_category", f"tyre_category must be one of {sorted(TYRE_CATEGORIES)}")
    meta = {"source": source, "metadata_origin": metadata_origin, "line_id": line_id, "camera_id": camera_id,
            "tyre_category": tyre_category, "notes": notes, "parent_image_id": str(parent_image_id or ""),
            "transform_hash": transform_hash, "scenario_id": scenario_id}
    raw_sha = sha256_bytes(raw) if raw else ""
    request_hash = sha256_json({"image": raw_sha, "meta": meta})

    existing = db.execute(select(Inspection).where(Inspection.created_by == actor_id,
                                                   Inspection.idempotency_key == idempotency_key)).scalar_one_or_none()
    if existing is not None:
        if existing.request_hash == request_hash:
            return existing, False
        raise ApiError(409, "idempotency_conflict", "This Idempotency-Key was used with a different payload.",
                       {"inspection_id": str(existing.id)})

    if enforce_backpressure:
        active = db.execute(select(func.count()).select_from(Job).where(
            Job.kind == "inference", Job.status.in_(("queued", "running", "retry_wait")))).scalar_one()
        if active >= get_settings().inference_queue_limit:
            raise ApiError(429, "queue_full", "Inspection queue is full; retry shortly.",
                           {"queued": active, "limit": get_settings().inference_queue_limit}, retry_after=5)

    _ptr, dep = active_deployment(db)
    if dep is None:
        raise ApiError(503, "model_unavailable", "No trained model available. Promote a validated bundle first.",
                       retry_after=30)
    # Pin the deployment tuple at acceptance, then release the pooled connection: image processing is CPU-bound
    # and must not hold a database connection (a burst of uploads would otherwise exhaust the pool).
    dep_id, dep_generation, pinned = dep.id, dep.generation, model_tuple(dep)
    if db.new or db.dirty or db.deleted:
        raise RuntimeError("accept() needs a session without pending changes; commit them first")
    db.rollback()  # nothing of the caller's is pending (checked above); this only returns the connection

    try:
        ing = ingest(raw, _limits(), PreprocessContract())
    except ImageRejected as exc:
        raise ApiError(exc.http_status, exc.code, exc.message, retryable=False) from exc
    except Exception as exc:  # Ultralytics patches PIL.Image.open (HEIF fallback), so decoders can raise anything
        raise ApiError(415, "unsupported_media", "Not a decodable image.", retryable=False) from exc

    # Find-or-create the image asset; object bytes are written before the SQL transaction commits.
    if parent_image_id is None:
        asset = db.execute(select(ImageAsset).where(ImageAsset.original_sha256 == raw_sha,
                                                    ImageAsset.parent_image_id.is_(None))).scalar_one_or_none()
    else:
        asset = db.execute(select(ImageAsset).where(ImageAsset.parent_image_id == parent_image_id,
                                                    ImageAsset.transform_hash == transform_hash)).scalar_one_or_none()
    if asset is None:
        key = (f"uploads/{raw_sha}.jpg" if parent_image_id is None else
               f"derivatives/{parent_image_id}/{transform_hash}.jpg")
        try:
            storage.put(key, ing.sanitized, "image/jpeg")
        except StorageUnavailable as exc:
            raise ApiError(503, "storage_unavailable", "Object storage is unavailable; retry shortly.",
                           retry_after=10) from exc
        parent = db.get(ImageAsset, parent_image_id) if parent_image_id else None
        asset = ImageAsset(
            original_sha256=raw_sha, sanitized_sha256=ing.info["sanitized_sha256"], parent_image_id=parent_image_id,
            transform=transform, transform_hash=transform_hash,
            group_id=parent.group_id if parent else None,
            source_id=parent.source_id if parent else "upload",
            label=parent.label if parent else None,
            label_origin=parent.label_origin if parent else "pending",
            outer_split=parent.outer_split if parent else None,
            role=parent.role if parent else None,
            object_key=key, width=ing.info["width"], height=ing.info["height"],
            image_origin="synthetic_derivative" if parent else "user_upload",
            metadata_origin=metadata_origin, training_eligible=False, ingest_info=ing.info,
        )
        db.add(asset)
    inspection_id = uuid.uuid4()
    job = jobs.create_job(db, kind="inference", dedupe_key=f"inference:{inspection_id}",
                          payload={"inspection_id": str(inspection_id)}, actor_id=actor_id,
                          resource_type="inspection", resource_id=str(inspection_id))
    insp = Inspection(
        id=inspection_id, image=asset, job_id=job.id, deployment_id=dep_id, deployment_generation=dep_generation,
        model_tuple=pinned, status="queued", source=source, line_id=line_id, camera_id=camera_id,
        scenario_id=scenario_id, replay_job_id=replay_job_id, replay_sequence=replay_sequence,
        metadata_origin=metadata_origin, tyre_category=tyre_category, notes=notes,
        idempotency_key=idempotency_key, request_hash=request_hash, created_by=actor_id,
    )
    db.add(insp)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()  # concurrent duplicate: return the winner if it matches, else 409
        winner = db.execute(select(Inspection).where(Inspection.created_by == actor_id,
                                                     Inspection.idempotency_key == idempotency_key)).scalar_one_or_none()
        if winner is not None and winner.request_hash == request_hash:
            return winner, False
        if winner is not None:
            raise ApiError(409, "idempotency_conflict",
                           "This Idempotency-Key was used with a different payload.") from exc
        raise
    return insp, True


def accepted_body(insp: Inspection) -> dict[str, Any]:
    return {
        "inspection_id": str(insp.id), "job_id": str(insp.job_id), "status": insp.status,
        "deployment_id": str(insp.deployment_id), "deployment_generation": insp.deployment_generation,
        "metadata_origin": insp.metadata_origin, "poll_after_ms": 1500,
        "status_url": f"/v1/inspections/{insp.id}",
    }


@lru_cache(maxsize=32)
def _bundle_policy(bundle_sha: str) -> dict[str, Any] | None:
    """Policy thresholds of an immutable bundle (for predictions stored before lineage carried them)."""
    from ..storage import get_storage

    try:
        pol = json.loads(get_storage().get(f"bundles/{bundle_sha}/policy.json"))
    except Exception:
        return None
    return {k: pol.get(k) for k in ("version", "binary_threshold", "triage_low", "triage_high")}


def _review_dict(r: ReviewLabel, reviewer: User | None) -> dict[str, Any]:
    return {"revision": r.revision, "label": r.label, "disposition": r.disposition, "reason": r.reason,
            "reviewer": reviewer.display_name if reviewer else None, "verified_at": r.verified_at.isoformat()}


def detail(db: Session, insp: Inspection, include_timeline: bool = True) -> dict[str, Any]:
    pred = db.execute(select(Prediction).where(Prediction.inspection_id == insp.id)).scalar_one_or_none()
    reviews = db.execute(select(ReviewLabel, User).join(User, User.id == ReviewLabel.reviewer_id)
                         .where(ReviewLabel.inspection_id == insp.id).order_by(ReviewLabel.revision)).all()
    stages = db.execute(select(InspectionStage).where(InspectionStage.inspection_id == insp.id)
                        .order_by(InspectionStage.stage_index)).scalars().all()
    job = db.get(Job, insp.job_id)
    img = insp.image
    out: dict[str, Any] = {
        "inspection_id": str(insp.id),
        "status": insp.status,
        "created_at": insp.created_at.isoformat(),
        "completed_at": insp.completed_at.isoformat() if insp.completed_at else None,
        "source": insp.source, "line_id": insp.line_id, "camera_id": insp.camera_id,
        "scenario_id": insp.scenario_id, "replay_sequence": insp.replay_sequence,
        "metadata_origin": insp.metadata_origin, "tyre_category": insp.tyre_category, "notes": insp.notes,
        "image": {
            "image_id": str(img.id), "url": sign_media(img.object_key), "width": img.width, "height": img.height,
            "original_sha256": img.original_sha256, "image_origin": img.image_origin,
            "parent_image_id": str(img.parent_image_id) if img.parent_image_id else None,
            "transform": img.transform, "label_origin": img.label_origin,
            "publisher_label": img.label if img.label_origin == "publisher" else None,
            "role": img.role,
        },
        "deployment": insp.model_tuple,
        "prediction": None,
        "review": {"revision": reviews[-1][0].revision if reviews else 0,
                   "label": reviews[-1][0].label if reviews else None,
                   "disposition": reviews[-1][0].disposition if reviews else None},
        "review_history": [_review_dict(r, u) for r, u in reviews],
        "stages": [
            {"index": s.stage_index, "name": s.name, "title": s.title, "duration_ms": round(s.duration_ms, 2),
             "params": s.params, "log": s.log, "image_url": sign_media(s.object_key) if s.object_key else None}
            for s in stages
        ],
        "job": {
            "job_id": str(job.id), "status": job.status, "attempt": job.attempt, "max_attempts": job.max_attempts,
            "progress": job.progress, "error_code": job.error_code,
            "error_message": job.error_message if job.status in ("failed", "retry_wait") else None,
            "next_retry_at": job.next_retry_at.isoformat() if job.next_retry_at else None,
        } if job else None,
    }
    if pred is not None:
        out["prediction"] = {
            "task": "classify", "raw_probs": pred.raw_probs, "calibrated_probs": pred.calibrated_probs,
            "p_defective": pred.p_defective, "triage": pred.triage, "triage_reason": pred.triage_reason,
            "binary_defective": pred.binary_defective, "quality_status": pred.quality_status,
            "quality": pred.quality, "timing": pred.timing, "lineage": pred.lineage,
            "attribution": pred.attribution, "circularity": pred.circularity,
            "policy": pred.lineage.get("policy") or _bundle_policy(pred.lineage.get("bundle_sha256", "")),
            "heatmap_url": next((st["image_url"] for st in out["stages"] if st["name"] == "attribution"), None),
            "classified_image_url": next((st["image_url"] for st in out["stages"] if st["name"] == "classified"), None),
        }
    if include_timeline and job is not None:
        events = db.execute(select(JobEvent).where(JobEvent.job_id == job.id).order_by(JobEvent.id)).scalars().all()
        out["timeline"] = [{"at": e.at.isoformat(), "from": e.from_status, "to": e.to_status, "attempt": e.attempt,
                            "reason": e.reason} for e in events]
    return out


def summary(insp: Inspection, pred: Prediction | None, review: ReviewLabel | None) -> dict[str, Any]:
    return {
        "inspection_id": str(insp.id), "status": insp.status, "created_at": insp.created_at.isoformat(),
        "source": insp.source, "line_id": insp.line_id, "camera_id": insp.camera_id,
        "scenario_id": insp.scenario_id, "metadata_origin": insp.metadata_origin,
        "thumbnail_url": sign_media(insp.image.object_key),
        "triage": pred.triage if pred else None, "p_defective": pred.p_defective if pred else None,
        "binary_defective": pred.binary_defective if pred else None,
        "quality_status": pred.quality_status if pred else None,
        "human_label": review.label if review else None, "disposition": review.disposition if review else None,
        "model_version": insp.model_tuple.get("registry_version"),
        "bundle_sha256": insp.model_tuple.get("bundle_sha256"),
        "deployment_generation": insp.deployment_generation,
    }


# ---- reviews ----------------------------------------------------------------------------------------------

LABELS = {"good", "defective", "unjudgeable"}
DISPOSITIONS = {"retread", "repair", "material_recovery", "reinspect"}


def append_review(db: Session, *, actor: Principal, inspection_id: uuid.UUID, expected_revision: int, label: str,
                  reason: str, disposition: str | None, correlation_id: str | None) -> dict[str, Any]:
    if label not in LABELS:
        raise ApiError(422, "invalid_label", f"label must be one of {sorted(LABELS)}")
    if disposition is not None and disposition not in DISPOSITIONS:
        raise ApiError(422, "invalid_disposition", f"disposition must be one of {sorted(DISPOSITIONS)}")
    if not reason or not reason.strip():
        raise ApiError(422, "reason_required", "A review reason is required.")
    insp = db.execute(select(Inspection).where(Inspection.id == inspection_id)
                      .with_for_update(of=Inspection)).scalar_one_or_none()  # lock only the inspection row
    if insp is None:
        raise ApiError(404, "not_found", "Inspection not found.")
    current = db.execute(select(func.coalesce(func.max(ReviewLabel.revision), 0))
                         .where(ReviewLabel.inspection_id == inspection_id)).scalar_one()
    if expected_revision != current:
        latest = db.execute(select(ReviewLabel).where(ReviewLabel.inspection_id == inspection_id,
                                                      ReviewLabel.revision == current)).scalar_one_or_none()
        db.rollback()
        raise ApiError(409, "stale_revision", "Another reviewer changed this label; reload and reconcile.",
                       {"expected_revision": expected_revision, "current_revision": current,
                        "latest": {"label": latest.label, "disposition": latest.disposition, "reason": latest.reason}
                        if latest else None})
    rev = ReviewLabel(inspection_id=inspection_id, revision=current + 1, label=label, disposition=disposition,
                      reason=reason.strip()[:2000], reviewer_id=actor.id)
    db.add(rev)
    audit.record(db, actor=actor, action="review.append", resource_type="inspection", resource_id=inspection_id,
                 correlation_id=correlation_id,
                 payload={"revision": rev.revision, "label": label, "disposition": disposition})
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ApiError(409, "stale_revision", "A concurrent review won; reload and reconcile.") from exc
    return {"inspection_id": str(inspection_id), "revision": rev.revision, "label": label, "disposition": disposition}
