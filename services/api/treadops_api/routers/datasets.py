from __future__ import annotations

import io
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit, jobs
from ..db import get_db
from ..errors import ApiError
from ..models import DatasetSnapshot
from ..security import Principal, require
from ..services import datasets as svc
from ..storage import get_storage

router = APIRouter(prefix="/v1", tags=["datasets"])


def snapshot_view(s: DatasetSnapshot) -> dict:
    return {"id": str(s.id), "manifest_sha256": s.manifest_sha256, "dvc_rev": s.dvc_rev,
            "source_versions": s.source_versions, "split_policy": s.split_policy, "counts": s.counts,
            "license_refs": s.license_refs, "description": s.description, "created_at": s.created_at.isoformat(),
            "job_id": str(s.job_id) if s.job_id else None}


@router.get("/datasets")
def datasets(p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    snaps = db.execute(select(DatasetSnapshot).order_by(DatasetSnapshot.created_at.desc()).limit(50)).scalars().all()
    return {**svc.catalogue(db), "snapshots": [snapshot_view(s) for s in snaps]}


class SnapshotIn(BaseModel):
    description: str | None = Field(default=None, max_length=500)
    include_verified_uploads: bool = True


@router.post("/datasets/snapshots", status_code=202)
def create_snapshot(request: Request, body: SnapshotIn | None = None, p: Principal = Depends(require("ml_engineer")),
                    db: Session = Depends(get_db)):
    body = body or SnapshotIn()
    job = jobs.create_job(db, kind="snapshot", dedupe_key=f"snapshot:{uuid.uuid4()}", payload=body.model_dump(),
                          actor_id=p.id, correlation_id=request.state.request_id, resource_type="dataset_snapshot")
    audit.record(db, actor=p, action="dataset.snapshot.requested", resource_type="job", resource_id=job.id,
                 correlation_id=request.state.request_id, payload=body.model_dump())
    db.commit()
    return {"job_id": str(job.id), "status": "queued"}


@router.get("/datasets/{snapshot_id}/manifest")
def manifest(snapshot_id: uuid.UUID, role: str | None = None, label: str | None = None,
             offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100),
             p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    snap = db.get(DatasetSnapshot, snapshot_id)
    if snap is None:
        raise ApiError(404, "not_found", "Snapshot not found.")
    key = snap.artifact_uri.removeprefix(f"s3://{get_storage().bucket}/")
    df = pd.read_parquet(io.BytesIO(get_storage().get(key)))
    if role:
        df = df[df.role == role]
    if label:
        df = df[df.label == label]
    page = df.iloc[offset : offset + limit]
    cols = ["image_id", "group_id", "label", "label_origin", "role", "outer_split", "image_origin", "metadata_origin",
            "original_sha256"]
    return {"snapshot_id": str(snap.id), "manifest_sha256": snap.manifest_sha256, "total": int(len(df)),
            "offset": offset, "limit": limit, "items": page[cols].to_dict("records")}
