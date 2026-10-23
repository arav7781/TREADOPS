"""Casing circularity ledger and per-inspection casing passport."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import and_, func, select, text
from sqlalchemy.orm import Session

from .. import circularity
from ..db import get_db
from ..errors import ApiError
from ..models import Inspection, Prediction, ReviewLabel
from ..security import Principal, require
from ..services.inspections import detail

router = APIRouter(prefix="/v1", tags=["circularity"])


def _latest_reviews():
    latest = (select(ReviewLabel.inspection_id, func.max(ReviewLabel.revision).label("rev"))
              .group_by(ReviewLabel.inspection_id).subquery())
    return (select(ReviewLabel).join(latest, and_(latest.c.inspection_id == ReviewLabel.inspection_id,
                                                  latest.c.rev == ReviewLabel.revision))).subquery()


@router.get("/circularity/ledger")
def ledger(since: datetime | None = None, metadata_origin: str | None = None,
           p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    since = since or datetime.now(UTC) - timedelta(days=365)
    conds = [Inspection.created_at >= since]
    if metadata_origin:
        conds.append(Inspection.metadata_origin == metadata_origin)
    rv = _latest_reviews()
    dispositions: dict[str, int] = dict(db.execute(select(rv.c.disposition, func.count()).join(
        Inspection, Inspection.id == rv.c.inspection_id).where(*conds, rv.c.disposition.is_not(None))
        .group_by(rv.c.disposition)).all())
    pathway = Prediction.circularity["pathway"].astext.label("pathway")  # group by the label: one bound key
    recommended: dict[str, int] = dict(db.execute(
        select(pathway, func.count())
        .join(Inspection, Inspection.id == Prediction.inspection_id).where(*conds)
        .group_by(text("pathway"))).all())
    agreement = db.execute(select(Prediction.triage, rv.c.disposition, func.count())
                           .join(rv, rv.c.inspection_id == Prediction.inspection_id)
                           .join(Inspection, Inspection.id == Prediction.inspection_id)
                           .where(*conds, rv.c.disposition.is_not(None))
                           .group_by(Prediction.triage, rv.c.disposition)).all()
    by_origin = dict(db.execute(select(Inspection.metadata_origin, func.count()).join(
        rv, rv.c.inspection_id == Inspection.id).where(*conds, rv.c.disposition == "retread")
        .group_by(Inspection.metadata_origin)).all())
    out = circularity.ledger(dispositions, recommended)
    out["model_vs_inspector"] = [{"model_triage": t, "inspector_disposition": d, "count": n} for t, d, n in agreement]
    out["verified_retreads_by_metadata_origin"] = by_origin
    out["since"] = since.isoformat()
    return out


@router.get("/circularity/passport/{inspection_id}")
def passport(inspection_id: uuid.UUID, p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    """A casing passport: one inspection's decision, evidence, lineage and verified disposition."""
    insp = db.get(Inspection, inspection_id)
    if insp is None:
        raise ApiError(404, "not_found", "Inspection not found.")
    d = detail(db, insp, include_timeline=False)
    pred = d["prediction"] or {}
    review = d["review"]
    return {
        "passport_id": f"TOPS-{str(insp.id)[:8].upper()}",
        "inspection_id": str(insp.id),
        "issued_at": datetime.now(UTC).isoformat(),
        "casing": {"tyre_category": insp.tyre_category, "image_sha256": d["image"]["original_sha256"],
                   "image_url": d["image"]["url"], "metadata_origin": insp.metadata_origin},
        "model_assessment": {"triage": pred.get("triage"), "p_defective": pred.get("p_defective"),
                             "quality_status": pred.get("quality_status"),
                             "recommended_pathway": (pred.get("circularity") or {}).get("label"),
                             "classified_image_url": pred.get("classified_image_url")},
        "inspector_verification": {"label": review["label"], "disposition": review["disposition"],
                                   "revision": review["revision"],
                                   "history": d["review_history"]},
        "impact_estimate": (pred.get("circularity") or {}).get("estimate_if_verified")
        if review["disposition"] == "retread" else None,
        "lineage": pred.get("lineage"),
        "status": "verified" if review["disposition"] else "awaiting_inspector",
        "disclaimer": "Model output is a triage aid. Roadworthiness and retread eligibility require a qualified "
                      "inspector and the retreader's full casing inspection (e.g. shearography).",
    }
