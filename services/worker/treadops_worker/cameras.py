"""Camera connector: one reader thread per enabled camera turns live frames into normal inspections.

Each reader keeps the stream drained (so frames are current), stores a live snapshot every few seconds, and once
per `interval_s` takes the sharpest frame of that interval. The frame is inspected when it is sharp and well
exposed and, in `on_change` mode, when the scene changed since the last capture (a tyre arrived or turned), so a
stationary tyre or an empty belt is not classified over and over. There is no tyre detector: the change rule is a
heuristic and is documented as such. Inspections go through the normal acceptance path (pinned deployment,
backpressure, idempotency), so camera results get the same lineage, review and monitoring as uploads.

With an inspection zone (ROI) only that crop is judged and inspected. About 5 times a second the reader also
publishes an annotated live frame (overlay.py): the zone in the colour of the latest verdict and the damage
locator's boxes on that frame. Video files (file://, local demos only) are played at their own frame rate and loop.

Run: python -m treadops_worker.cameras   (scripts/dev.sh up cameras)
"""

from __future__ import annotations

import logging
import signal
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import cv2
import numpy as np
from sqlalchemy import select

from treadops_api import cameras as svc
from treadops_api.db import new_session
from treadops_api.errors import ApiError
from treadops_api.models import Camera, ModelVersion
from treadops_api.services.inspections import accept, active_deployment
from treadops_api.storage import get_storage

from . import overlay
from .runtime import bundles

log = logging.getLogger("treadops.cameras")
CONFIG_POLL_S = 5.0
STATUS_EVERY_S = 2.0
MIN_BRIGHTNESS, MAX_BRIGHTNESS = 0.08, 0.97
BACKOFF_MAX_S = 60.0
LIVE_EVERY_S = 0.2        # annotated live frames per second: 5
STILL_DIFF = 0.012        # mean grey change between consecutive zone thumbnails below which the zone is "still"
STILL_S = 0.4             # on_change: a tyre is captured once it has been still in the zone this long
RESULT_EVERY_S = 0.5      # how often the latest verdict is re-read for the overlay
RESULT_FRESH_S = 20.0     # a verdict older than this is not drawn (the tyre has moved on)


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


def is_still(previous: np.ndarray | None, current: np.ndarray) -> bool:
    return previous is not None and float(np.abs(current - previous).mean()) < STILL_DIFF


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
                "url_enc": c.stream_url_enc, "roi": c.roi, "overlay": c.overlay}
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


class LiveModel:
    """The serving bundle's damage locator for the live overlay; follows promotions (checked every 30 s)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._checked = 0.0
        self._bundle: Any = None

    def get(self) -> Any:
        with self._lock:
            if time.monotonic() - self._checked < 30:
                return self._bundle
            self._checked = time.monotonic()
            db = new_session()
            try:
                _, dep = active_deployment(db)
                mv = db.get(ModelVersion, dep.model_version_id) if dep else None
                if mv is not None and (self._bundle is None or self._bundle.sha != mv.bundle_sha256):
                    self._bundle = bundles().get(mv.bundle_sha256, mv.bundle_uri)
            except Exception as exc:  # the overlay is optional: live frames go out without boxes
                log.warning("live overlay: no bundle (%s)", exc)
            finally:
                db.close()
            return self._bundle


LIVE_MODEL = LiveModel()


def latest_result(code: str) -> dict[str, Any] | None:
    db = new_session()
    try:
        r = svc.latest_results(db, [code]).get(code)
    finally:
        db.close()
    if r is None or datetime.fromisoformat(r["at"]) < datetime.now(UTC) - timedelta(seconds=RESULT_FRESH_S):
        return None
    return r


def publish_live(cam: dict[str, Any], frame: np.ndarray, fps: float | None, result: dict[str, Any] | None,
                 inspecting: bool = False) -> None:
    """One annotated (or, with the overlay off, plain) live frame for the browser. `result` is drawn only when it
    is the verdict for the tyre now in the zone (the caller checks); `inspecting` while that verdict is pending."""
    boxes, experimental, model = None, False, None
    if cam["overlay"]:
        bundle = LIVE_MODEL.get()
        if bundle is not None:
            model = bundle.manifest.get("model_family")
            zone = svc.crop_roi(frame, cam["roi"])
            found = bundle.locate_damage(np.ascontiguousarray(zone[:, :, ::-1]))
            if found:
                boxes, experimental = found["boxes"], found.get("authority") == "display_only"
        frame = overlay.annotate(frame, roi=cam["roi"], result=result, boxes=boxes, experimental=experimental,
                                 camera=cam["name"], line=cam["line_id"], fps=fps, model=model,
                                 simulated=cam["simulated"], inspecting=inspecting)
    ok, jpg = cv2.imencode(".jpg", svc._fit(frame, 1280), [cv2.IMWRITE_JPEG_QUALITY, 82])
    if ok:
        svc.publish_live(cam["id"], jpg.tobytes(), {
            "at": datetime.now(UTC).isoformat(), "boxes": boxes or [], "experimental": experimental,
            "result": result, "inspecting": inspecting, "roi": cam["roi"]})


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
                self._stream(cap, cam, is_file=svc.decrypt_url(cam["url_enc"]).startswith("file://"))
            except Exception as exc:  # a broken stream must never take the connector down
                log.exception("camera %s reader failed", cam["name"])
                _status(self.camera_id, state="error", last_error=f"{exc.__class__.__name__}: {str(exc)[:200]}")
            finally:
                cap.release()
            self.stop.wait(backoff)

    def _stream(self, cap: Any, cam: dict[str, Any], is_file: bool = False) -> None:
        storage = get_storage()
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_dt = 1.0 / src_fps if is_file and 0 < src_fps < 240 else 0.0  # play files in real time
        next_frame = time.monotonic()
        last_live = last_result_read = 0.0
        result: dict[str, Any] | None = None
        submitted: tuple[str, np.ndarray] | None = None  # (inspection id, zone thumbnail) of the latest capture
        prev_thumb: np.ndarray | None = None
        still_since: float | None = None
        last_capture = 0.0
        fps: float | None = None
        frames = 0
        window_start = last_status = time.monotonic()
        fps_frames, fps_t0 = 0, time.monotonic()
        best: tuple[float, np.ndarray, np.ndarray, float] | None = None
        last_thumb: np.ndarray | None = None
        captures = 0
        _status(self.camera_id, state="streaming", last_error=None)
        while not self.stop.is_set():
            if frame_dt:
                next_frame += frame_dt
                self.stop.wait(max(0.0, next_frame - time.monotonic()))
            ok, frame = cap.read()
            if (not ok or frame is None) and is_file:  # a demo video: start again from the first frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                next_frame = time.monotonic()
                ok, frame = cap.read()
            if not ok or frame is None:
                _status(self.camera_id, state="error", last_error="stream ended or no frame within the read timeout")
                return
            frames += 1
            fps_frames += 1
            thumb, sharp, bright = frame_metrics(svc.crop_roi(frame, cam["roi"]))
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
            if now - last_result_read >= RESULT_EVERY_S:
                result, last_result_read = latest_result(cam["code"]), now
            if now - last_live >= LIVE_EVERY_S:
                last_live = now
                # The verdict belongs to the latest capture, and only while that tyre is still in the zone.
                same_tyre = submitted is not None and not scene_changed(submitted[1], thumb, cam["change_threshold"])
                shown = result if same_tyre and result and submitted and result["inspection_id"] == submitted[0] else None
                try:
                    publish_live(cam, frame, fps, shown, inspecting=same_tyre and shown is None)
                except Exception as exc:  # the live view is best effort; capture and inspection go on
                    log.warning("camera %s: live frame not published: %s", cam["name"], exc)
            # on_change: capture a tyre once it has stopped in the zone (a stop-and-go line, or a tyre put in front
            # of the camera) and differs from the last capture; interval: the sharpest frame of each interval
            # (a belt that never stops).
            still_since = (still_since or now) if is_still(prev_thumb, thumb) else None
            prev_thumb = thumb
            due = None
            if cam["capture_mode"] == "on_change":
                if still_since is not None and now - still_since >= STILL_S and now - last_capture >= cam["interval_s"]:
                    due = (sharp, frame, thumb, bright)
            elif now - window_start >= cam["interval_s"] and best is not None:
                due = best
            if due is not None:
                sharp_b, frame_b, thumb_b, bright_b = due
                capture, reason = should_capture(cam, sharp_b, bright_b, last_thumb, thumb_b)
                inspection_id = submit(cam, svc.crop_roi(frame_b, cam["roi"]), sharp_b) if capture else None
                last_capture = now if inspection_id else last_capture
                if inspection_id:
                    submitted = (inspection_id, thumb_b)
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
