"""Camera connector: one reader thread per enabled camera turns live frames into normal inspections.

Each reader keeps the stream drained (so frames are current), stores a live snapshot every few seconds, and once
per `interval_s` takes the sharpest frame of that interval. The frame is inspected when it is sharp and well
exposed and, in `on_change` mode, when the scene changed since the last capture (a tyre arrived or turned), so a
stationary tyre or an empty belt is not classified over and over. There is no tyre detector: the change rule is a
heuristic and is documented as such. Inspections go through the normal acceptance path (pinned deployment,
backpressure, idempotency), so camera results get the same lineage, review and monitoring as uploads.

Run: python -m treadops_worker.cameras   (scripts/dev.sh up cameras)
"""

from __future__ import annotations

import logging
import signal
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np
from sqlalchemy import select

from treadops_api import cameras as svc
from treadops_api.db import new_session
from treadops_api.errors import ApiError
from treadops_api.models import Camera
from treadops_api.services.inspections import accept
from treadops_api.storage import get_storage

log = logging.getLogger("treadops.cameras")
CONFIG_POLL_S = 5.0
STATUS_EVERY_S = 2.0
MIN_BRIGHTNESS, MAX_BRIGHTNESS = 0.08, 0.97
BACKOFF_MAX_S = 60.0


def frame_metrics(bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(64x64 grey thumbnail in [0, 1], sharpness as Laplacian variance at <=320 px, mean brightness in [0, 1])."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape
    scale = min(1.0, 320 / max(h, w))
    small = cv2.resize(grey, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
    thumb = cv2.resize(grey, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return thumb, sharp, float(thumb.mean())


def scene_changed(previous: np.ndarray | None, current: np.ndarray, threshold: float) -> bool:
    return previous is None or float(np.abs(current - previous).mean()) > threshold


def should_capture(cam: dict[str, Any], sharp: float, bright: float, previous: np.ndarray | None,
                   current: np.ndarray) -> tuple[bool, str]:
    """The capture rule, kept pure for tests: returns (capture?, reason)."""
    if sharp < cam["min_sharpness"]:
        return False, f"too blurred (sharpness {sharp:.1f} < {cam['min_sharpness']})"
    if not MIN_BRIGHTNESS <= bright <= MAX_BRIGHTNESS:
        return False, f"badly exposed (brightness {bright:.2f})"
    if cam["capture_mode"] == "on_change" and not scene_changed(previous, current, cam["change_threshold"]):
        return False, "scene unchanged since the last capture"
    return True, "captured"


def _load(camera_id: uuid.UUID) -> dict[str, Any] | None:
    db = new_session()
    try:
        c = db.get(Camera, camera_id)
        if c is None:
            return None
        return {"id": c.id, "name": c.name, "code": c.code, "line_id": c.line_id, "enabled": c.enabled,
                "simulated": c.simulated, "capture_mode": c.capture_mode, "interval_s": c.interval_s,
                "change_threshold": c.change_threshold, "min_sharpness": c.min_sharpness,
                "url_enc": c.stream_url_enc}
    finally:
        db.close()


def _status(camera_id: uuid.UUID, **fields: Any) -> None:
    db = new_session()
    try:
        c = db.get(Camera, camera_id)
        if c is not None:
            for k, v in fields.items():
                setattr(c, k, v)
            db.commit()
    finally:
        db.close()


def submit(cam: dict[str, Any], frame: np.ndarray, sharp: float) -> str | None:
    """Inspect one frame through the normal acceptance path. Returns the inspection id, or None if refused."""
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return None
    db = new_session()
    try:
        insp, _ = accept(db, get_storage(), actor=None, actor_id=None, raw=jpg.tobytes(),
                         idempotency_key=f"camera:{cam['id']}:{time.time_ns()}", source="camera",
                         metadata_origin="simulated" if cam["simulated"] else "observed", line_id=cam["line_id"],
                         camera_id=cam["code"], notes=f"captured by {cam['name']} (sharpness {sharp:.0f})")
        return str(insp.id)
    except ApiError as exc:  # queue full (429), no active model, rejected image...
        log.warning("camera %s: frame not inspected: %s", cam["name"], exc.code)
        return None
    finally:
        db.close()


class Reader(threading.Thread):
    def __init__(self, camera_id: uuid.UUID):
        super().__init__(name=f"camera-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.stop = threading.Event()

    def run(self) -> None:
        backoff = 2.0
        while not self.stop.is_set():
            cam = _load(self.camera_id)
            if cam is None or not cam["enabled"]:
                return
            _status(self.camera_id, state="connecting")
            try:
                cap = svc.open_stream(svc.decrypt_url(cam["url_enc"]))
            except ApiError as exc:
                _status(self.camera_id, state="error", last_error=exc.message)
                self.stop.wait(backoff)
                backoff = min(BACKOFF_MAX_S, backoff * 2)
                continue
            backoff = 2.0
            try:
                self._stream(cap, cam)
            except Exception as exc:  # a broken stream must never take the connector down
                log.exception("camera %s reader failed", cam["name"])
                _status(self.camera_id, state="error", last_error=f"{exc.__class__.__name__}: {str(exc)[:200]}")
            finally:
                cap.release()
            self.stop.wait(backoff)

    def _stream(self, cap: Any, cam: dict[str, Any]) -> None:
        storage = get_storage()
        frames = 0
        window_start = last_status = time.monotonic()
        fps_frames, fps_t0 = 0, time.monotonic()
        best: tuple[float, np.ndarray, np.ndarray, float] | None = None
        last_thumb: np.ndarray | None = None
        captures = 0
        _status(self.camera_id, state="streaming", last_error=None)
        while not self.stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                _status(self.camera_id, state="error", last_error="stream ended or no frame within the read timeout")
                return
            frames += 1
            fps_frames += 1
            thumb, sharp, bright = frame_metrics(frame)
            if best is None or sharp > best[0]:
                best = (sharp, frame, thumb, bright)
            now = time.monotonic()
            if now - last_status >= STATUS_EVERY_S:
                ok_jpg, jpg = cv2.imencode(".jpg", svc._fit(frame, 960), [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok_jpg:
                    storage.put(svc.snapshot_key(self.camera_id), jpg.tobytes(), "image/jpeg")
                fps = fps_frames / max(now - fps_t0, 1e-6)
                fps_frames, fps_t0 = 0, now
                _status(self.camera_id, state="streaming", last_frame_at=datetime.now(UTC), fps=round(fps, 1),
                        frame_width=int(frame.shape[1]), frame_height=int(frame.shape[0]),
                        frames_read=Camera.frames_read + frames)
                frames = 0
                last_status = now
                fresh = _load(self.camera_id)  # pick up edits (interval, thresholds) and disabling without a restart
                if fresh is None or not fresh["enabled"]:
                    return
                cam = {**fresh}
            if now - window_start >= cam["interval_s"] and best is not None:
                sharp_b, frame_b, thumb_b, bright_b = best
                capture, reason = should_capture(cam, sharp_b, bright_b, last_thumb, thumb_b)
                if capture and submit(cam, frame_b, sharp_b):
                    last_thumb = thumb_b
                    captures += 1
                    _status(self.camera_id, last_capture_at=datetime.now(UTC), captures=Camera.captures + 1)
                else:
                    log.debug("camera %s: skipped (%s)", cam["name"], reason)
                best, window_start = None, now


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    readers: dict[uuid.UUID, Reader] = {}
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    log.info("camera connector started")
    while not stopping.is_set():
        db = new_session()
        try:
            enabled = set(db.execute(select(Camera.id).where(Camera.enabled.is_(True))).scalars().all())
        finally:
            db.close()
        for cid in enabled - {c for c, r in readers.items() if r.is_alive()}:
            readers[cid] = Reader(cid)
            readers[cid].start()
            log.info("camera %s: reader started", cid)
        for cid in [c for c in readers if c not in enabled]:
            readers.pop(cid).stop.set()
            log.info("camera %s: reader stopped", cid)
        stopping.wait(CONFIG_POLL_S)
    for r in readers.values():
        r.stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
