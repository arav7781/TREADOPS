"""Append-only audit trail. Written inside the same transaction as the action it records."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from .models import AuditEvent


def record(
    db: Session,
    *,
    actor=None,
    action: str,
    resource_type: str,
    resource_id: Any = None,
    correlation_id: str | None = None,
    outcome: str = "success",
    payload: dict[str, Any] | None = None,
) -> AuditEvent:
    ev = AuditEvent(
        actor_id=getattr(actor, "id", None),
        actor_email=getattr(actor, "email", None),
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id is not None else None,
        correlation_id=correlation_id,
        outcome=outcome,
        payload=payload or {},
    )
    db.add(ev)
    return ev
