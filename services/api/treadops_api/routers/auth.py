from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit
from ..db import get_db
from ..errors import ApiError
from ..models import User, UserSession
from ..security import Principal, current_principal, issue_token, verify_password

router = APIRouter(prefix="/v1", tags=["auth"])


class LoginIn(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=200)


@router.post("/auth/login")
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)):
    user = db.execute(select(User).where(User.email == body.email.strip().lower())).scalar_one_or_none()
    if user is None or not user.is_active or not verify_password(body.password, user.password_hash):
        audit.record(db, action="auth.login", resource_type="user", outcome="denied",
                     payload={"email_hash": hashlib.sha256(body.email.lower().encode()).hexdigest()[:16]},
                     correlation_id=request.state.request_id)
        db.commit()
        raise ApiError(401, "invalid_credentials", "Email or password is incorrect.")
    token, exp = issue_token(db, user)
    audit.record(db, actor=user, action="auth.login", resource_type="user", resource_id=user.id,
                 correlation_id=request.state.request_id)
    db.commit()
    return {"access_token": token, "token_type": "bearer", "expires_at": exp.isoformat(),
            "user": {"id": str(user.id), "email": user.email, "display_name": user.display_name, "roles": user.roles}}


@router.post("/auth/logout")
def logout(p: Principal = Depends(current_principal), db: Session = Depends(get_db)):
    sess = db.get(UserSession, p.session_id)
    if sess and sess.revoked_at is None:
        sess.revoked_at = datetime.now(UTC)
    db.commit()
    return {"status": "signed_out"}


@router.get("/me")
def me(p: Principal = Depends(current_principal)):
    return {"id": str(p.id), "email": p.email, "display_name": p.display_name, "roles": p.roles,
            "allowed_actions": p.allowed_actions}
