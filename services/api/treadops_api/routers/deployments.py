from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Deployment, User
from ..security import Principal, require
from ..services import releases
from ..services.inspections import active_deployment

router = APIRouter(prefix="/v1", tags=["deployments"])


class ReleaseIn(BaseModel):
    model_version_id: uuid.UUID
    expected_generation: int = Field(ge=0)
    reason: str = Field(min_length=8, max_length=1000)


@router.get("/deployments")
def history(p: Principal = Depends(require("viewer")), db: Session = Depends(get_db)):
    ptr, dep = active_deployment(db)
    db.commit()
    rows = db.execute(select(Deployment, User).outerjoin(User, User.id == Deployment.approved_by)
                      .order_by(Deployment.generation.desc())).all()
    return {
        "generation": ptr.generation,
        "active_deployment_id": str(dep.id) if dep else None,
        "items": [{"deployment_id": str(d.id), "generation": d.generation, "action": d.action, "state": d.state,
                   "model_version_id": str(d.model_version_id), "registry_version": d.model_version.registry_version,
                   "bundle_sha256": d.bundle_sha256, "reason": d.reason,
                   "approved_by": u.display_name if u else None, "readiness": d.readiness,
                   "previous_deployment_id": str(d.previous_deployment_id) if d.previous_deployment_id else None,
                   "created_at": d.created_at.isoformat()} for d, u in rows],
    }


@router.post("/deployments/promote", status_code=202)
def promote(body: ReleaseIn, request: Request, x_correlation_id: str | None = Header(None),
            p: Principal = Depends(require("admin")), db: Session = Depends(get_db)):
    return releases.request_release(db, actor=p, action="promote", model_version_id=body.model_version_id,
                                    expected_generation=body.expected_generation, reason=body.reason,
                                    correlation_id=x_correlation_id or request.state.request_id)


@router.post("/deployments/rollback", status_code=202)
def rollback(body: ReleaseIn, request: Request, x_correlation_id: str | None = Header(None),
             p: Principal = Depends(require("admin")), db: Session = Depends(get_db)):
    return releases.request_release(db, actor=p, action="rollback", model_version_id=body.model_version_id,
                                    expected_generation=body.expected_generation, reason=body.reason,
                                    correlation_id=x_correlation_id or request.state.request_id)
