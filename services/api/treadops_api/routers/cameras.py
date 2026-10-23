"""Line cameras: register RTSP (or HTTP) streams, test them, watch them and inspect what they see."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit
from .. import cameras as svc
from ..db import get_db
from ..errors import ApiError
from ..models import Camera
from ..security import Principal, require, sign_media
from ..services.inspections import accept, accepted_body
from ..storage import get_storage

router = APIRouter(prefix="/v1", tags=["cameras"])
SNAPSHOT_MAX_AGE_S = 30


class CameraIn(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    stream_url: str = Field(min_length=8, max_length=1000)
    line_id: str | None = Field(None, max_length=64)
    simulated: bool = Field(False, description="Test stream (e.g. a looped video): its inspections are labelled simulated")
    enabled: bool = True
    capture_mode: str = Field("on_change", pattern="^(interval|on_change)$")
    interval_s: float = Field(5.0, ge=1, le=3600)
    change_threshold: float = Field(0.04, ge=0, le=1)
    min_sharpness: float = Field(15.0, ge=0, le=100000)


class CameraPatch(BaseModel):
    name: str | None = Field(None, min_length=2, max_length=80)
    stream_url: str | None = Field(None, min_length=8, max_length=1000)
    line_id: str | None = Field(None, max_length=64)
    simulated: bool | None = None
    enabled: bool | None = None
    capture_mode: str | None = Field(None, pattern="^(interval|on_change)$")
    interval_s: float | None = Field(None, ge=1, le=3600)
    change_threshold: float | None = Field(None, ge=0, le=1)
    min_sharpness: float | None = Field(None, ge=0, le=100000)


class ProbeIn(BaseModel):
    stream_url: str = Field(min_length=8, max_length=1000)


def _get(db: Session, camera_id: uuid.UUID) -> Camera:
    c = db.get(Camera, camera_id)
    if c is None:
        raise ApiError(404, "not_found", "Camera not found.")
    return c


@router.get("/cameras")
def list_cameras(p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    cams = db.execute(select(Camera).order_by(Camera.name)).scalars().all()
    st = svc.stats(db, [c.code for c in cams])
    return {"items": [svc.view(c, st.get(c.code)) for c in cams]}


@router.post("/cameras/probe")
async def probe(body: ProbeIn, p: Principal = Depends(require("ml_engineer"))):
    """Connect to a stream before saving it; returns one frame as a short-lived signed snapshot."""
    url = svc.validate_url(body.stream_url)
    jpg, facts = await run_in_threadpool(svc.probe, url)
    key = get_storage().put(f"cameras/probe/{uuid.uuid4()}.jpg", jpg, "image/jpeg")
    return {"ok": True, **facts, "snapshot_url": sign_media(key), "stream_url": svc.mask_url(url)}


@router.post("/cameras", status_code=201)
def create(body: CameraIn, request: Request, p: Principal = Depends(require("ml_engineer")),
           db: Session = Depends(get_db)):
    url = svc.validate_url(body.stream_url)
    if db.execute(select(Camera).where(Camera.name == body.name)).scalar_one_or_none():
        raise ApiError(409, "camera_exists", "A camera with this name already exists.")
    code = svc.slug(body.name)
    if db.execute(select(Camera).where(Camera.code == code)).scalar_one_or_none():
        code = f"{code}-{uuid.uuid4().hex[:4]}"
    c = Camera(name=body.name, code=code, line_id=body.line_id, stream_url_enc=svc.encrypt_url(url),
               stream_url_masked=svc.mask_url(url), simulated=body.simulated, enabled=body.enabled,
               capture_mode=body.capture_mode, interval_s=body.interval_s, change_threshold=body.change_threshold,
               min_sharpness=body.min_sharpness, state="idle" if body.enabled else "disabled", created_by=p.id)
    db.add(c)
    db.flush()
    audit.record(db, actor=p, action="camera.create", resource_type="camera", resource_id=c.id,
                 correlation_id=request.state.request_id,
                 payload={"name": c.name, "line_id": c.line_id, "stream_url": c.stream_url_masked,
                          "simulated": c.simulated})
    db.commit()
    return svc.view(c)


@router.patch("/cameras/{camera_id}")
def update(camera_id: uuid.UUID, body: CameraPatch, request: Request, p: Principal = Depends(require("ml_engineer")),
           db: Session = Depends(get_db)):
    c = _get(db, camera_id)
    changes = body.model_dump(exclude_unset=True)
    if "stream_url" in changes:
        url = svc.validate_url(changes.pop("stream_url"))
        c.stream_url_enc, c.stream_url_masked = svc.encrypt_url(url), svc.mask_url(url)
        changes["stream_url"] = c.stream_url_masked
        c.state = "idle"
    for k, v in changes.items():
        if k != "stream_url":
            setattr(c, k, v)
    if "enabled" in changes:
        c.state = "idle" if c.enabled else "disabled"
    audit.record(db, actor=p, action="camera.update", resource_type="camera", resource_id=c.id,
                 correlation_id=request.state.request_id, payload=changes)
    db.commit()
    return svc.view(c)


@router.delete("/cameras/{camera_id}", status_code=204)
def delete(camera_id: uuid.UUID, request: Request, p: Principal = Depends(require("ml_engineer")),
           db: Session = Depends(get_db)):
    c = _get(db, camera_id)
    audit.record(db, actor=p, action="camera.delete", resource_type="camera", resource_id=c.id,
                 correlation_id=request.state.request_id, payload={"name": c.name})
    db.delete(c)  # its inspections stay: they keep camera_id = the camera's code
    db.commit()


@router.get("/cameras/{camera_id}/snapshot")
def snapshot(camera_id: uuid.UUID, p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    c = _get(db, camera_id)
    if c.last_frame_at is None:
        raise ApiError(404, "no_snapshot", "The connector has not read a frame from this camera yet.")
    return {"url": sign_media(svc.snapshot_key(c.id)), "at": c.last_frame_at.isoformat()}


def inspect_now(db: Session, c: Camera, actor: Principal | None, request_key: str) -> dict:
    """Inspect the camera's latest frame (kept fresh by the connector) as a normal pinned inspection."""
    if c.last_frame_at is None or c.last_frame_at < datetime.now(UTC) - timedelta(seconds=SNAPSHOT_MAX_AGE_S):
        raise ApiError(409, "camera_not_live", f"{c.name} has no frame from the last {SNAPSHOT_MAX_AGE_S} s; "
                                               "is the connector running and the stream up?")
    raw = get_storage().get(svc.snapshot_key(c.id))
    insp, _ = accept(db, get_storage(), actor=actor, actor_id=getattr(actor, "id", None), raw=raw,
                     idempotency_key=request_key, source="camera",
                     metadata_origin="simulated" if c.simulated else "observed", line_id=c.line_id,
                     camera_id=c.code, notes=f"on-demand frame from {c.name}")
    return accepted_body(insp)


@router.post("/cameras/{camera_id}/inspect", status_code=202)
def inspect(camera_id: uuid.UUID, request: Request, p: Principal = Depends(require("inspector")),
            db: Session = Depends(get_db)):
    c = _get(db, camera_id)
    return inspect_now(db, c, p, request.headers.get("idempotency-key") or f"camera-now:{uuid.uuid4()}")
