"""FastAPI application: authoritative /v1 contract. Routers validate and delegate; they never train."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import errors
from .routers import (
    audit,
    auth,
    cameras,
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
        description="Tyre casing inspection with image classification (YOLO26 or ConvNeXt bundles), a governed MLOps release loop and a "
        "guarded circularity copilot. All times UTC ISO 8601; all IDs UUID.",
        openapi_url="/v1/openapi.json",
        docs_url="/v1/docs",
        redoc_url=None,
    )
    errors.install(app)
    # Admission control: image uploads are parsed and re-encoded in-process, so a burst saturates the API before
    # any inference job is queued. Cap in-flight uploads per process and refuse the excess before reading bodies.
    upload_paths = {"/v1/inspections", "/v1/copilot/messages"}
    upload_cap = int(os.environ.get("MAX_INFLIGHT_UPLOADS", "8"))
    inflight = {"uploads": 0}

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        is_upload = request.method == "POST" and request.url.path in upload_paths
        if is_upload and inflight["uploads"] >= upload_cap:
            access.info(json.dumps({"request_id": rid, "path": request.url.path, "status": 429,
                                    "reason": "upload_admission", "inflight": inflight["uploads"]}))
            return JSONResponse({"error": {"code": "too_many_uploads", "message": "Upload capacity is busy; retry "
                                           "shortly.", "request_id": rid, "retryable": True,
                                           "details": {"inflight": inflight["uploads"], "limit": upload_cap}}},
                                status_code=429, headers={"Retry-After": "2", "X-Request-ID": rid})
        if is_upload:
            inflight["uploads"] += 1
        try:
            response = await call_next(request)
        finally:
            if is_upload:
                inflight["uploads"] -= 1
        response.headers["X-Request-ID"] = rid
        p = getattr(request.state, "principal", None)
        access.info(json.dumps({"request_id": rid, "method": request.method, "path": request.url.path,
                                "status": response.status_code, "ms": round((time.perf_counter() - t0) * 1000, 1),
                                "user": str(p.id) if p else None}))
        return response

    for r in (health, auth, inspections, media, jobs, datasets, training, models, deployments, monitoring, replays,
              audit, metrics, circularity, copilot, llmops, cameras):
        app.include_router(r.router)
    return app


logging.basicConfig(level=logging.INFO, format="%(message)s")
app = create_app()
