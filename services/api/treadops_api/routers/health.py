"""Liveness and readiness. Readiness reports an explicit no-model state instead of pretending."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import redis
from alembic.config import Config
from alembic.script import ScriptDirectory
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import select, text

from treadops_ml.common import repo_path

from ..db import session_scope
from ..models import WorkerHeartbeat
from ..services.inspections import active_deployment
from ..settings import get_settings
from ..storage import get_storage

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz():  # event loop, no threadpool and no database: liveness only
    return {"status": "ok", "at": datetime.now(UTC).isoformat()}


def _head() -> str | None:
    return ScriptDirectory.from_config(Config(str(repo_path("alembic.ini")))).get_current_head()


@router.get("/readyz")
def readyz():
    checks: dict = {}
    model: dict = {"state": "unavailable"}
    try:
        with session_scope() as db:
            db.execute(text("SELECT 1"))
            checks["sql"] = "ok"
            current = db.execute(text("SELECT version_num FROM alembic_version")).scalar()
            checks["migrations"] = "ok" if current == _head() else f"behind ({current})"
            ptr, dep = active_deployment(db)
            fresh = datetime.now(UTC) - timedelta(seconds=45)
            beats = db.execute(select(WorkerHeartbeat).where(WorkerHeartbeat.last_seen_at > fresh)).scalars().all()
            inference = [b for b in beats if "inference" in b.queue]
            checks["inference_worker"] = "ok" if inference else "no heartbeat"
            if dep is None:
                model = {"state": "untrained", "message": "No trained model available"}
            else:
                ready = any(dep.bundle_sha256 in (b.ready_bundles or []) for b in inference)
                model = {"state": "ready" if ready else "loading", "bundle_sha256": dep.bundle_sha256,
                         "generation": dep.generation, "registry_version": dep.model_version.registry_version}
    except Exception as exc:
        checks["sql"] = f"error: {exc.__class__.__name__}"
    try:
        redis.Redis.from_url(get_settings().redis_url, socket_timeout=2).ping()
        checks["broker"] = "ok"
    except Exception as exc:
        checks["broker"] = f"error: {exc.__class__.__name__}"
    checks["storage"] = "ok" if get_storage().ping() else "error"
    ready = all(v == "ok" for v in checks.values()) and model["state"] == "ready"
    return JSONResponse({"ready": ready, "checks": checks, "model": model}, status_code=200 if ready else 503)
