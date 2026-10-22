"""Uniform error contract: {"error": {code, message, request_id, retryable, details}}. No stack traces leak."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("treadops.api")

RETRYABLE = {429, 502, 503, 504}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict[str, Any] | None = None,
                 retryable: bool | None = None, retry_after: int | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}
        self.retryable = status in RETRYABLE if retryable is None else retryable
        self.retry_after = retry_after


def _body(request: Request, status: int, code: str, message: str, details=None, retryable=False):
    return {"error": {"code": code, "message": message, "request_id": getattr(request.state, "request_id", None),
                      "retryable": retryable, "details": details or {}}}


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return JSONResponse(_body(request, exc.status, exc.code, exc.message, exc.details, exc.retryable),
                            status_code=exc.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        details = {"fields": [{"loc": [str(x) for x in e.get("loc", [])], "msg": e.get("msg")} for e in exc.errors()]}
        return JSONResponse(_body(request, 422, "validation_error", "Request failed validation.", details), 422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return JSONResponse(_body(request, exc.status_code, code, str(exc.detail)), exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error request_id=%s", getattr(request.state, "request_id", None))
        return JSONResponse(_body(request, 500, "internal_error", "Unexpected server error.", retryable=True), 500)
