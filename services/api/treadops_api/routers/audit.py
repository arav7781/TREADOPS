from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import AuditEvent
from ..security import Principal, require

router = APIRouter(prefix="/v1", tags=["audit"])


@router.get("/audit-events")
def audit_events(
    action: str | None = Query(None, description="prefix match, e.g. release."),
    resource_type: str | None = None,
    actor: str | None = None,
    outcome: str | None = None,
    since: datetime | None = None,
    before_id: int | None = Query(None, description="cursor: return events with id < before_id"),
    limit: int = Query(50, ge=1, le=100),
    p: Principal = Depends(require("admin")),
    db: Session = Depends(get_db),
):
    q = select(AuditEvent)
    if action:
        q = q.where(AuditEvent.action.startswith(action))
    if resource_type:
        q = q.where(AuditEvent.resource_type == resource_type)
    if actor:
        q = q.where(AuditEvent.actor_email == actor)
    if outcome:
        q = q.where(AuditEvent.outcome == outcome)
    if since:
        q = q.where(AuditEvent.at >= since)
    if before_id:
        q = q.where(AuditEvent.id < before_id)
    rows = db.execute(q.order_by(AuditEvent.id.desc()).limit(limit + 1)).scalars().all()
    items = [{"id": e.id, "at": e.at.isoformat(), "actor": e.actor_email, "action": e.action,
              "resource_type": e.resource_type, "resource_id": e.resource_id, "outcome": e.outcome,
              "correlation_id": e.correlation_id, "payload": e.payload} for e in rows[:limit]]
    return {"items": items, "next_before_id": rows[limit - 1].id if len(rows) > limit else None}
