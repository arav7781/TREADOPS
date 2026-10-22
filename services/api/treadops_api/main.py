"""FastAPI application: authoritative /v1 contract. Routers validate and delegate; they never train."""

from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import FastAPI, Request

from . import errors
from .routers import (
    audit,
    auth,
    circularity,
    copilot,
    datasets,
    deployments,
    health,
    inspections,
    jobs,
    llmops,
    media,
    metrics,
    models,
    monitoring,
    replays,
    training,
)

access = logging.getLogger("treadops.access")


def create_app() -> FastAPI:
    app = FastAPI(
        title="TreadOps API",
        version="1.0.0",
        description="Tyre casing inspection with YOLO26 classification, a governed MLOps release loop and a "
        "guarded circularity copilot. All times UTC ISO 8601; all IDs UUID.",
        openapi_url="/v1/openapi.json",
        docs_url="/v1/docs",
        redoc_url=None,
    )
    errors.install(app)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        p = getattr(request.state, "principal", None)
        access.info(json.dumps({"request_id": rid, "method": request.method, "path": request.url.path,
                                "status": response.status_code, "ms": round((time.perf_counter() - t0) * 1000, 1),
                                "user": str(p.id) if p else None}))
        return response

    for r in (health, auth, inspections, media, jobs, datasets, training, models, deployments, monitoring, replays,
              audit, metrics, circularity, copilot, llmops):
        app.include_router(r.router)
    return app


logging.basicConfig(level=logging.INFO, format="%(message)s")
app = create_app()
