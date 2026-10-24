"""DatasetService: public-catalogue registration and immutable training-eligible snapshots.

Eligibility (logged per reason):
  public images (every source) in dev_train/dev_val roles (frozen split)         -> included
  inspector-verified uploads (good/defective), originals only, own new group       -> included
  calibration / policy / test roles and anything descended from them              -> excluded (protected)
  replay / synthetic derivatives, unjudgeable or unreviewed uploads               -> excluded
"""

from __future__ import annotations

import io
import json
import uuid
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from treadops_ml.common import read_json, repo_path, sha256_bytes, sha256_json
from treadops_ml.descriptors import hamming, phash
from treadops_ml.preprocess import load_canonical

from .. import audit
from ..errors import ApiError
from ..models import DatasetSnapshot, ImageAsset, Inspection, ReviewLabel
from ..storage import Storage

PROTECTED_ROLES = ("calibration", "policy", "test")
D1_LICENSE = "CC BY 4.0 - Pathmanaban P. et al., Mendeley Data V1, doi:10.17632/bn7ch8tvyp.1"


def register_public_catalogue(db: Session, storage: Storage, *, upload: bool = True) -> dict[str, Any]:
    """Load the frozen D1 manifests into image_assets and upload canonical bytes (idempotent)."""
    man = pd.read_parquet(repo_path("data/manifests/image_manifest.parquet"))
    split = pd.read_parquet(repo_path("data/manifests/split_assignments.parquet")).set_index("image_id")
    ok = man[man.status == "ok"]
    existing = {str(r) for r in db.execute(select(ImageAsset.id).where(ImageAsset.image_origin == "public_dataset"))
                .scalars()}
    created = uploaded = 0
    for r in ok.itertuples():
        if r.image_id in existing:
            continue
        s = split.loc[r.image_id]
        key = f"images/{r.original_sha256}.jpg"
        if upload and not storage.exists(key):
            storage.put(key, repo_path(f"data/processed/images/{r.original_sha256}.jpg").read_bytes(), "image/jpeg")
            uploaded += 1
        db.add(ImageAsset(
            id=uuid.UUID(r.image_id), original_sha256=r.original_sha256, sanitized_sha256=None,
            canonical_sha256=r.canonical_sha256, group_id=r.group_id, source_id=r.source_id, label=r.label,
            label_origin="publisher", outer_split=s.outer_split, role=s.role, object_key=key, canonical_key=key,
            width=int(r.width), height=int(r.height), image_origin="public_dataset", metadata_origin="observed",
            training_eligible=s.role in ("dev_train", "dev_val"),
            ingest_info={"format": r.source_format, "original_width": int(r.original_width),
                         "original_height": int(r.original_height), "member": r.member},
        ))
        created += 1
    db.flush()
    return {"created": created, "uploaded": uploaded, "total_public": len(ok)}


def _latest_review_labels(db: Session):
    latest = (select(ReviewLabel.inspection_id, func.max(ReviewLabel.revision).label("rev"))
              .group_by(ReviewLabel.inspection_id).subquery())
    return db.execute(
        select(Inspection.image_id, ReviewLabel.label, ReviewLabel.verified_at)
        .join(latest, latest.c.inspection_id == Inspection.id)
        .join(ReviewLabel, and_(ReviewLabel.inspection_id == latest.c.inspection_id,
                                ReviewLabel.revision == latest.c.rev))
    ).all()


def build_snapshot(db: Session, storage: Storage, *, actor_id=None, job_id=None, description: str | None = None,
                   include_verified_uploads: bool = True) -> tuple[DatasetSnapshot, bool]:
    rows: list[dict[str, Any]] = []
    excluded: dict[str, int] = {}
    public = db.execute(select(ImageAsset).where(ImageAsset.image_origin == "public_dataset")).scalars().all()
    protected_groups = {a.group_id for a in public if a.role in PROTECTED_ROLES}
    for a in public:
        if a.role in ("dev_train", "dev_val"):
            rows.append(_row(a, a.label, "publisher", a.role))
        else:
            excluded[f"protected_role:{a.role}"] = excluded.get(f"protected_role:{a.role}", 0) + 1

    n_derivatives = db.execute(select(func.count()).select_from(ImageAsset)
                               .where(ImageAsset.parent_image_id.is_not(None))).scalar_one()
    if n_derivatives:
        excluded["replay_or_synthetic_derivative"] = n_derivatives

    if include_verified_uploads:
        labels: dict[uuid.UUID, tuple[str, datetime]] = {}
        for image_id, label, verified_at in _latest_review_labels(db):
            labels.setdefault(image_id, (label, verified_at))
        uploads = db.execute(select(ImageAsset).where(ImageAsset.image_origin == "user_upload",
                                                      ImageAsset.parent_image_id.is_(None))).scalars().all()
        man = pd.read_parquet(repo_path("data/manifests/image_manifest.parquet"), columns=["image_id", "phash"])
        protected_ids = {str(a.id) for a in public if a.role in PROTECTED_ROLES}
        protected_hashes = [int(h, 16) for i, h in zip(man.image_id, man.phash, strict=True)
                            if i in protected_ids and isinstance(h, str)]
        for a in uploads:
            lab = labels.get(a.id)
            if lab is None:
                excluded["upload_unreviewed"] = excluded.get("upload_unreviewed", 0) + 1
            elif lab[0] == "unjudgeable":
                excluded["upload_unjudgeable"] = excluded.get("upload_unjudgeable", 0) + 1
            elif a.group_id in protected_groups:
                excluded["upload_in_protected_group"] = excluded.get("upload_in_protected_group", 0) + 1
            elif not a.canonical_key:
                excluded["upload_not_processed"] = excluded.get("upload_not_processed", 0) + 1
            elif any(hamming(phash(load_canonical(storage.get(a.canonical_key))), h) <= 6 for h in protected_hashes):
                excluded["upload_near_duplicate_of_protected"] = excluded.get("upload_near_duplicate_of_protected", 0) + 1
            else:
                role = "dev_val" if int(sha256_bytes(str(a.id).encode())[:8], 16) % 100 < 15 else "dev_train"
                rows.append(_row(a, lab[0], "inspector_verified", role, group=f"upl-{a.original_sha256[:16]}"))

    if not rows:
        raise ApiError(409, "empty_snapshot", "No training-eligible images: register the dataset catalogue first.",
                       {"excluded": excluded})
    rows.sort(key=lambda r: r["image_id"])
    manifest_sha = sha256_json(rows)
    existing = db.execute(select(DatasetSnapshot).where(DatasetSnapshot.manifest_sha256 == manifest_sha)
                          ).scalar_one_or_none()
    if existing is not None:
        return existing, False
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    key = f"snapshots/{manifest_sha}/manifest.parquet"
    storage.put(key, buf.getvalue())
    split_manifest = read_json("data/manifests/split_manifest.json")
    counts = {
        "included": len(rows),
        "by_role_label": {f"{r}:{lab}": int(n) for (r, lab), n in df.groupby(["role", "label"]).size().items()},
        "by_label_origin": {k: int(v) for k, v in df.label_origin.value_counts().items()},
        "excluded": excluded,
        "groups": int(df.group_id.nunique()),
    }
    snap = DatasetSnapshot(
        manifest_sha256=manifest_sha, dvc_rev=_dvc_rev(), source_versions={"D1": "v1", "uploads": "verified"},
        split_policy={"frozen_split_sha256": split_manifest["split_sha256"], "protected_roles": list(PROTECTED_ROLES),
                      "upload_assignment": "sha256(image_id) % 100 < 15 -> dev_val"},
        counts=counts, artifact_uri=f"s3://{storage.bucket}/{key}", parent_ids=[], license_refs=[D1_LICENSE],
        description=description, created_by=actor_id, job_id=job_id,
    )
    db.add(snap)
    db.flush()
    audit.record(db, actor=None, action="dataset.snapshot.created", resource_type="dataset_snapshot",
                 resource_id=snap.id, payload={"manifest_sha256": manifest_sha, "counts": counts})
    return snap, True


def _row(a: ImageAsset, label: str | None, label_origin: str, role: str,
         group: str | None = None) -> dict[str, Any]:
    return {"image_id": str(a.id), "group_id": group or a.group_id, "label": label, "label_origin": label_origin,
            "role": role, "outer_split": "train", "image_origin": a.image_origin,
            "metadata_origin": a.metadata_origin, "canonical_key": a.canonical_key,
            "original_sha256": a.original_sha256}


def _dvc_rev() -> str | None:
    lock = repo_path("dvc.lock")
    return sha256_bytes(lock.read_bytes())[:16] if lock.exists() else None


def catalogue(db: Session) -> dict[str, Any]:
    src = repo_path("data/manifests/source_manifest.json")
    q = repo_path("data/manifests/quality_report.json")
    split = repo_path("data/manifests/split_manifest.json")
    roles = db.execute(select(ImageAsset.role, ImageAsset.label, func.count(), func.count(func.distinct(ImageAsset.group_id)))
                       .where(ImageAsset.image_origin == "public_dataset")
                       .group_by(ImageAsset.role, ImageAsset.label)).all()
    origins = dict(db.execute(select(ImageAsset.image_origin, func.count()).group_by(ImageAsset.image_origin)).all())
    return {
        "source": json.loads(src.read_text()) if src.exists() else None,
        "quality": json.loads(q.read_text()) if q.exists() else None,
        "split": json.loads(split.read_text()) if split.exists() else None,
        "roles": [{"role": r, "label": lab, "images": n, "groups": g} for r, lab, n, g in roles],
        "assets_by_origin": origins,
    }
