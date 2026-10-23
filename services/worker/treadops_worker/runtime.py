"""Per-process runtime: hash-verified bundle cache, reference encoder, worker identity and heartbeats."""

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import threading
import time
from collections import OrderedDict
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert

from treadops_api.db import session_scope
from treadops_api.models import WorkerHeartbeat, utcnow
from treadops_api.settings import get_settings
from treadops_api.storage import get_storage
from treadops_ml.bundle import Bundle, BundleInvalid
from treadops_ml.common import repo_path

WORKER_ID = os.environ.get("WORKER_NAME") or f"{socket.gethostname()}:{os.getpid()}"


class BundleCache:
    """Small LRU of loaded bundles keyed by bundle SHA. Bundles are immutable, so a loaded object is never
    mutated; a promotion only changes which SHA new inspections pin."""

    def __init__(self, capacity: int = 3):
        self.capacity = capacity
        self._items: OrderedDict[str, Bundle] = OrderedDict()
        self._lock = threading.Lock()
        self.root = repo_path(get_settings().bundle_cache_dir)

    def local_dir(self, sha: str, uri: str) -> Path:
        d = self.root / sha
        if (d / "manifest.json").exists():
            return d
        tmp = self.root / f".{sha}.{os.getpid()}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        prefix = uri.removeprefix(f"s3://{get_settings().s3_bucket}/").rstrip("/") + "/"
        files = get_storage().download_prefix(prefix, tmp)
        if not files:
            raise BundleInvalid("missing_bundle", f"no objects under {uri}")
        try:
            tmp.rename(d)
        except OSError:  # another process won the race; theirs is identical (content-addressed)
            shutil.rmtree(tmp, ignore_errors=True)
        return d

    def get(self, sha: str, uri: str, device: str | None = None) -> Bundle:
        with self._lock:
            if sha in self._items:
                self._items.move_to_end(sha)
                return self._items[sha]
        bundle = Bundle(self.local_dir(sha, uri), device=device or get_settings().inference_device, verify=True)
        if bundle.sha != sha:
            raise BundleInvalid("bundle_sha_mismatch", f"expected {sha}, loaded {bundle.sha}")
        with self._lock:
            self._items[sha] = bundle
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return bundle

    def loaded(self) -> list[str]:
        with self._lock:
            return list(self._items)


_cache: BundleCache | None = None
_encoder = None
_enc_lock = threading.Lock()


def bundles() -> BundleCache:
    global _cache
    if _cache is None:
        _cache = BundleCache()
    return _cache


def reference_encoder():
    """The fixed drift encoder (pinned pretrained checkpoint), independent of the served model."""
    global _encoder
    with _enc_lock:
        if _encoder is None:
            from treadops_ml.descriptors import ReferenceEncoder
            from treadops_ml.weights import pretrained

            _encoder = ReferenceEncoder(str(pretrained("yolo26n-cls.pt")), device="cpu")
    return _encoder


def beat(queue: str, paused: bool = False, info: dict | None = None) -> None:
    with session_scope() as db:
        stmt = insert(WorkerHeartbeat).values(worker_name=WORKER_ID, queue=queue, last_seen_at=utcnow(),
                                              ready_bundles=bundles().loaded(), paused=paused, info=info or {})
        db.execute(stmt.on_conflict_do_update(
            index_elements=["worker_name"],
            set_={"last_seen_at": utcnow(), "ready_bundles": bundles().loaded(), "queue": queue, "paused": paused,
                  "info": info or {}}))


def prewarm_active() -> str | None:
    """Load the bundle the serving pointer names, so readiness reflects a warm model before traffic arrives."""
    from treadops_api.services.inspections import active_deployment

    with session_scope() as db:
        _ptr, dep = active_deployment(db)
        if dep is None:
            return None
        sha, uri = dep.bundle_sha256, dep.model_version.bundle_uri
    if sha not in bundles().loaded():
        b = bundles().get(sha, uri)
        # Ultralytics builds its predictor on the first predict(); run one golden image so the first real
        # request is not paying the setup cost.
        golden = sorted((b.root / "golden").glob("*.jpg"))
        if golden:
            from treadops_ml.preprocess import tensor_from_canonical

            b.raw_probs(tensor_from_canonical(golden[0].read_bytes(), b.preprocess))
    return sha


def start_heartbeat(queue: str, interval: float = 10.0) -> threading.Thread:
    def loop():
        while True:
            if "inference" in queue:
                with contextlib.suppress(Exception):
                    prewarm_active()
            with contextlib.suppress(Exception):
                beat(queue, info={"pid": os.getpid()})
            time.sleep(interval)

    t = threading.Thread(target=loop, name="treadops-heartbeat", daemon=True)
    t.start()
    return t
