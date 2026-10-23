"""Line cameras: encrypted stream URLs, connection probing, live snapshots and per-camera quality statistics.

The camera connector process (`treadops_worker.cameras`) reads the streams and turns captured frames into normal
inspections (source `camera`, `camera_id` = the camera's code), so every camera result has the same lineage,
review and monitoring path as an uploaded photo.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from .errors import ApiError
from .models import Camera, Inspection, Prediction
from .settings import get_settings

ALLOWED_SCHEMES = {"rtsp", "rtsps", "http", "https"}
OPEN_TIMEOUT_MS = 6000
READ_TIMEOUT_MS = 6000


def _fernet() -> Fernet:
    # Derived from JWT_SECRET: rotating that secret means re-entering camera URLs (documented in SECURITY.md).
    key = hashlib.sha256(b"treadops-camera-url:" + get_settings().jwt_secret.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(key))


def encrypt_url(url: str) -> str:
    return _fernet().encrypt(url.encode()).decode()


def decrypt_url(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise ApiError(409, "camera_url_unreadable",
                       "The stored stream URL cannot be decrypted (was JWT_SECRET rotated?); enter it again.") from exc


def mask_url(url: str) -> str:
    """Hide credentials: rtsp://admin:secret@10.0.0.5:554/s1 -> rtsp://admin:•••@10.0.0.5:554/s1."""
    parts = urlsplit(url)
    if parts.password is None and "@" not in parts.netloc:
        return url
    user = parts.username or ""
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{user}:•••@{host}{port}" if user else f"•••@{host}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))


def validate_url(url: str) -> str:
    url = (url or "").strip()
    scheme = urlsplit(url).scheme.lower()
    file_ok = os.environ.get("CAMERA_ALLOW_FILE_URLS") == "true"
    if scheme == "file" and file_ok:
        return url
    if scheme not in ALLOWED_SCHEMES or not urlsplit(url).hostname:
        raise ApiError(422, "invalid_stream_url",
                       "Use an rtsp://, rtsps://, http:// or https:// stream URL with a host.")
    return url


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "camera"


def open_stream(url: str):
    """Open a stream with hard timeouts (FFmpeg backend). Returns an opened cv2.VideoCapture or raises ApiError."""
    import cv2

    target = url[len("file://"):] if url.startswith("file://") else url
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS])
    if not cap.isOpened():
        cap.release()
        raise ApiError(422, "camera_unreachable", "Could not open the stream (check the URL, network and credentials).")
    return cap


def probe(url: str) -> tuple[bytes, dict[str, Any]]:
    """Connect, read one decodable frame and return it as JPEG plus stream facts."""
    import cv2

    cap = open_stream(url)
    try:
        frame = None
        for _ in range(30):  # some encoders need a keyframe before the first decodable frame
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
                break
        if frame is None:
            raise ApiError(422, "camera_no_frames", "Connected, but no decodable video frame arrived.")
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        ok, jpg = cv2.imencode(".jpg", _fit(frame, 960), [cv2.IMWRITE_JPEG_QUALITY, 88])
        return jpg.tobytes(), {"width": int(frame.shape[1]), "height": int(frame.shape[0]),
                               "fps": round(float(fps), 2) if fps and fps < 1000 else None}
    finally:
        cap.release()


def _fit(frame, side: int):
    import cv2

    h, w = frame.shape[:2]
    scale = min(1.0, side / max(h, w))
    return frame if scale == 1.0 else cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def snapshot_key(camera_id: Any) -> str:
    return f"cameras/{camera_id}/latest.jpg"


def stats(db: Session, codes: list[str], hours: int = 1) -> dict[str, dict[str, int]]:
    """Damaged / not damaged / expert-check counts per camera code over the last `hours`. The three add up: "not
    damaged" is a clean auto-clear only, and every other non-damaged tyre is an expert check (as in verdict_of)."""
    if not codes:
        return {}
    since = datetime.now(UTC) - timedelta(hours=hours)
    rows = db.execute(
        select(Inspection.camera_id,
               func.count(Prediction.inspection_id),
               func.sum(case((Prediction.binary_defective.is_(True), 1), else_=0)),
               func.sum(case((Prediction.binary_defective.is_(False) & (Prediction.triage == "good_candidate")
                              & (Prediction.quality_status != "low_quality"), 1), else_=0)))
        .join(Prediction, Prediction.inspection_id == Inspection.id)
        .where(Inspection.camera_id.in_(codes), Inspection.source == "camera", Inspection.created_at >= since)
        .group_by(Inspection.camera_id)).all()
    out = {c: {"inspected": 0, "damaged": 0, "not_damaged": 0, "expert_check": 0} for c in codes}
    for code, n, damaged, clear in rows:
        out[code] = {"inspected": int(n), "damaged": int(damaged or 0), "not_damaged": int(clear or 0),
                     "expert_check": int(n) - int(damaged or 0) - int(clear or 0)}
    return out


def view(c: Camera, st: dict[str, int] | None = None) -> dict[str, Any]:
    stale = c.last_frame_at is None or c.last_frame_at < datetime.now(UTC) - timedelta(seconds=30)
    return {"id": str(c.id), "name": c.name, "code": c.code, "line_id": c.line_id,
            "stream_url": c.stream_url_masked, "simulated": c.simulated, "enabled": c.enabled,
            "capture_mode": c.capture_mode, "interval_s": c.interval_s, "change_threshold": c.change_threshold,
            "min_sharpness": c.min_sharpness, "state": c.state, "last_error": c.last_error,
            "last_frame_at": c.last_frame_at.isoformat() if c.last_frame_at else None,
            "last_capture_at": c.last_capture_at.isoformat() if c.last_capture_at else None,
            "live": c.state == "streaming" and not stale, "fps": c.fps,
            "resolution": f"{c.frame_width}x{c.frame_height}" if c.frame_width else None,
            "frames_read": c.frames_read, "captures": c.captures,
            "last_hour": st or {"inspected": 0, "damaged": 0, "not_damaged": 0, "expert_check": 0}}
