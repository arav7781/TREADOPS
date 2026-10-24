"""Authentication, role checks and signed media URLs.

Passwords: salted scrypt (stdlib). Sessions: short-lived HS256 JWT with iss/aud/exp/roles plus a server-side
session row so logout revokes immediately. Roles are always re-read from the database, never from the token
alone and never from the browser.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session

from .db import get_db
from .errors import ApiError
from .models import User, UserSession
from .settings import get_settings

ROLE_RANK = {"viewer": 0, "inspector": 1, "ml_engineer": 1, "admin": 3}
ALL_ROLES = tuple(ROLE_RANK)

# scrypt parameters (n=2^14, r=8, p=1): interactive-login strength for a local prototype
_N, _R, _P, _DKLEN = 2**14, 8, 1, 32


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN)
    return f"scrypt${_N}${_R}${_P}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _algo, n, r, p, salt_b64, dk_b64 = stored.split("$")
        dk = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt_b64), n=int(n), r=int(r), p=int(p),
                            dklen=len(base64.b64decode(dk_b64)))
        return hmac.compare_digest(dk, base64.b64decode(dk_b64))
    except Exception:
        return False


@dataclass
class Principal:
    id: uuid.UUID
    email: str
    display_name: str
    roles: list[str]
    session_id: uuid.UUID

    def has(self, role: str) -> bool:
        if "admin" in self.roles:
            return True
        if role == "viewer":
            return bool(self.roles)
        return role in self.roles

    @property
    def allowed_actions(self) -> list[str]:
        acts = ["read"]
        if self.has("inspector"):
            acts += ["upload", "review", "copilot"]
        if self.has("ml_engineer"):
            acts += ["snapshot", "train", "replay", "shadow", "copilot"]
        if self.has("admin"):
            acts += ["promote", "rollback", "audit", "manage_users"]
        return sorted(set(acts))


def issue_token(db: Session, user: User) -> tuple[str, datetime]:
    s = get_settings()
    now = datetime.now(UTC)
    exp = now + timedelta(minutes=s.session_ttl_minutes)
    sess = UserSession(user_id=user.id, expires_at=exp)
    db.add(sess)
    db.flush()
    token = jwt.encode(
        {"sub": str(user.id), "sid": str(sess.id), "roles": user.roles, "iss": s.jwt_issuer, "aud": s.jwt_audience,
         "iat": int(now.timestamp()), "exp": int(exp.timestamp())},
        s.jwt_secret,
        algorithm="HS256",
    )
    return token, exp


def _decode(token: str) -> dict:
    s = get_settings()
    try:
        return jwt.decode(token, s.jwt_secret, algorithms=["HS256"], audience=s.jwt_audience, issuer=s.jwt_issuer,
                          options={"require": ["exp", "iat", "sub", "sid"]})
    except jwt.ExpiredSignatureError as exc:
        raise ApiError(401, "session_expired", "Session expired; sign in again.") from exc
    except jwt.PyJWTError as exc:
        raise ApiError(401, "invalid_token", "Authentication required.") from exc


def current_principal(
    request: Request, authorization: str | None = Header(None), db: Session = Depends(get_db)
) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise ApiError(401, "missing_token", "Authentication required.")
    claims = _decode(authorization.split(" ", 1)[1].strip())
    sess = db.get(UserSession, uuid.UUID(claims["sid"]))
    if sess is None or sess.revoked_at is not None or sess.expires_at < datetime.now(UTC):
        raise ApiError(401, "session_revoked", "Session is no longer valid.")
    user = db.get(User, uuid.UUID(claims["sub"]))
    if user is None or not user.is_active:
        raise ApiError(401, "user_inactive", "Account is not active.")
    p = Principal(user.id, user.email, user.display_name, list(user.roles), sess.id)
    request.state.principal = p
    return p


def require(role: str, *alternatives: str):
    """Any one of the roles grants access (e.g. require("inspector", "ml_engineer"))."""
    roles = (role, *alternatives)

    def dep(p: Principal = Depends(current_principal)) -> Principal:
        if not any(p.has(r) for r in roles):
            raise ApiError(403, "forbidden", f"This action requires the {' or '.join(roles)} role.")
        return p

    return dep


# ---- signed media URLs ----------------------------------------------------------------------------------


def _b64(d: bytes) -> str:
    return base64.urlsafe_b64encode(d).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def sign_media(key: str, ttl: int | None = None) -> str:
    s = get_settings()
    body = _b64(json.dumps({"k": key, "e": int(time.time()) + (ttl or s.media_url_ttl_seconds)}).encode())
    sig = _b64(hmac.new(s.jwt_secret.encode(), b"media:" + body.encode(), hashlib.sha256).digest()[:20])
    return f"/v1/media/{body}.{sig}"


def verify_media(token: str) -> str:
    s = get_settings()
    try:
        body, sig = token.split(".", 1)
        expected = _b64(hmac.new(s.jwt_secret.encode(), b"media:" + body.encode(), hashlib.sha256).digest()[:20])
        if not hmac.compare_digest(sig, expected):
            raise ValueError("bad signature")
        payload = json.loads(_unb64(body))
    except Exception as exc:
        raise ApiError(403, "invalid_media_token", "Media link is invalid.") from exc
    if payload["e"] < time.time():
        raise ApiError(403, "media_link_expired", "Media link expired; reload the page.")
    return payload["k"]
