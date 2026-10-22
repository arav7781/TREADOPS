from __future__ import annotations

import base64
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, Header, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, aliased

from ..db import get_db
from ..errors import ApiError
from ..models import Inspection, Prediction, ReviewLabel
from ..security import Principal, require
from ..services import inspections as svc
from ..settings import get_settings
from ..storage import get_storage

router = APIRouter(prefix="/v1", tags=["inspections"])
MAX_PAGE = 100


@router.post("/inspections", status_code=202)
def create_inspection(
    request: Request,
    file: UploadFile = File(...),
    source: str = Form("upload"),
    metadata_origin: str = Form("user_provided"),
    line_id: str | None = Form(None, max_length=40),
    camera_id: str | None = Form(None, max_length=40),
    tyre_category: str | None = Form(None, max_length=32),
    notes: str | None = Form(None, max_length=1000),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    p: Principal = Depends(require("inspector")),
    db: Session = Depends(get_db),
):
    limit = get_settings().max_upload_bytes
    raw = file.file.read(limit + 1)  # bounded read: never buffer more than the limit + 1 byte
    if len(raw) > limit:
        raise ApiError(413, "payload_too_large", f"Image exceeds {limit} bytes.")
    kwargs = dict(actor=p, actor_id=p.id, raw=raw, idempotency_key=idempotency_key or "", source=source,
                  metadata_origin=metadata_origin, line_id=line_id, camera_id=camera_id,
                  tyre_category=tyre_category, notes=notes)
    try:
        insp, created = svc.accept(db, get_storage(), **kwargs)
    except IntegrityError:  # lost a race creating the same image asset; the asset now exists
        db.rollback()
        insp, created = svc.accept(db, get_storage(), **kwargs)
    return JSONResponse(svc.accepted_body(insp), status_code=202,
                        headers={"Location": f"/v1/inspections/{insp.id}",
                                 "Idempotent-Replay": "false" if created else "true"})


def _cursor_encode(ts: datetime, iid: uuid.UUID) -> str:
    return base64.urlsafe_b64encode(f"{ts.isoformat()}|{iid}".encode()).decode()


def _cursor_decode(c: str) -> tuple[datetime, uuid.UUID]:
    try:
        ts, iid = base64.urlsafe_b64decode(c.encode()).decode().split("|")
        return datetime.fromisoformat(ts), uuid.UUID(iid)
    except Exception as exc:
        raise ApiError(400, "invalid_cursor", "Cursor is malformed.") from exc


@router.get("/inspections")
def list_inspections(
    limit: int = Query(25, ge=1, le=MAX_PAGE),
    cursor: str | None = None,
    status: str | None = None,
    triage: str | None = None,
    source: str | None = None,
    camera_id: str | None = None,
    line_id: str | None = None,
    scenario_id: str | None = None,
    model: str | None = Query(None, description="registry version or bundle sha prefix"),
    metadata_origin: str | None = None,
    reviewed: bool | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    p: Principal = Depends(require("viewer")),
    db: Session = Depends(get_db),
):
    latest = (select(ReviewLabel.inspection_id, func.max(ReviewLabel.revision).label("rev"))
              .group_by(ReviewLabel.inspection_id).subquery())
    rl = aliased(ReviewLabel)
    q = (select(Inspection, Prediction, rl)
         .outerjoin(Prediction, Prediction.inspection_id == Inspection.id)
         .outerjoin(latest, latest.c.inspection_id == Inspection.id)
         .outerjoin(rl, and_(rl.inspection_id == Inspection.id, rl.revision == latest.c.rev)))
    conds = []
    if status:
        conds.append(Inspection.status == status)
    if triage:
        conds.append(Prediction.triage == triage)
    if source:
        conds.append(Inspection.source == source)
    if camera_id:
        conds.append(Inspection.camera_id == camera_id)
    if line_id:
        conds.append(Inspection.line_id == line_id)
    if scenario_id:
        conds.append(Inspection.scenario_id == scenario_id)
    if metadata_origin:
        conds.append(Inspection.metadata_origin == metadata_origin)
    if model:
        conds.append(or_(Inspection.model_tuple["registry_version"].astext == model,
                         Inspection.model_tuple["bundle_sha256"].astext.startswith(model)))
    if reviewed is not None:
        conds.append(rl.id.is_not(None) if reviewed else rl.id.is_(None))
    if since:
        conds.append(Inspection.created_at >= since)
    if until:
        conds.append(Inspection.created_at < until)
    if cursor:
        ts, iid = _cursor_decode(cursor)
        conds.append(or_(Inspection.created_at < ts, and_(Inspection.created_at == ts, Inspection.id < iid)))
    rows = db.execute(q.where(*conds).order_by(Inspection.created_at.desc(), Inspection.id.desc())
                      .limit(limit + 1)).all()
    items = [svc.summary(i, pr, r) for i, pr, r in rows[:limit]]
    nxt = _cursor_encode(rows[limit - 1][0].created_at, rows[limit - 1][0].id) if len(rows) > limit else None
    return {"items": items, "next_cursor": nxt, "limit": limit,
            "filters": {k: v for k, v in dict(status=status, triage=triage, source=source, camera_id=camera_id,
                                              line_id=line_id, scenario_id=scenario_id, model=model,
                                              metadata_origin=metadata_origin, reviewed=reviewed).items() if v is not None}}


@router.get("/inspections/{inspection_id}")
def get_inspection(inspection_id: uuid.UUID, p: Principal = Depends(require("viewer")),
                   db: Session = Depends(get_db)):
    insp = db.get(Inspection, inspection_id)
    if insp is None:
        raise ApiError(404, "not_found", "Inspection not found.")
    return svc.detail(db, insp)


class ReviewIn(BaseModel):
    expected_revision: int = Field(ge=0)
    label: str
    reason: str = Field(min_length=1, max_length=2000)
    disposition: str | None = None


@router.post("/inspections/{inspection_id}/reviews", status_code=201)
def add_review(inspection_id: uuid.UUID, body: ReviewIn, request: Request,
               x_correlation_id: str | None = Header(None), p: Principal = Depends(require("inspector")),
               db: Session = Depends(get_db)):
    return svc.append_review(db, actor=p, inspection_id=inspection_id, expected_revision=body.expected_revision,
                             label=body.label, reason=body.reason, disposition=body.disposition,
                             correlation_id=x_correlation_id or request.state.request_id)


@router.get("/reviews/queue")
def review_queue(
    sort: str = Query("p_defective_desc", pattern="^(p_defective_desc|p_defective_asc|oldest|newest)$"),
    triage: str | None = None,
    limit: int = Query(50, ge=1, le=MAX_PAGE),
    p: Principal = Depends(require("viewer")),
    db: Session = Depends(get_db),
):
    """Completed inspections without a human label: escalations and low-quality photos first by default."""
    has_review = select(ReviewLabel.id).where(ReviewLabel.inspection_id == Inspection.id).exists()
    q = (select(Inspection, Prediction).join(Prediction, Prediction.inspection_id == Inspection.id)
         .where(Inspection.status == "succeeded", ~has_review))
    if triage:
        q = q.where(Prediction.triage == triage)
    order = {"p_defective_desc": Prediction.p_defective.desc(), "p_defective_asc": Prediction.p_defective.asc(),
             "oldest": Inspection.created_at.asc(), "newest": Inspection.created_at.desc()}[sort]
    rows = db.execute(q.order_by(order, Inspection.id).limit(limit)).all()
    total = db.execute(select(func.count()).select_from(q.subquery())).scalar_one()
    oldest = db.execute(select(func.min(Inspection.created_at)).select_from(q.subquery())).scalar_one_or_none() \
        if total else None
    return {"items": [svc.summary(i, pr, None) for i, pr in rows], "total": total, "sort": sort,
            "oldest_waiting_at": oldest.isoformat() if oldest else None}
