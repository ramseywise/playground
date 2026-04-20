"""FastAPI middleware: request identity, timeout, and structured logging."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from interfaces.api.models import ErrorResponse

log = structlog.get_logger(__name__)

_HEALTH_PATHS = frozenset({"/health", "/api/v1/health"})


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique trace_id to every request.

    Reads X-Request-ID from the incoming request (propagated by the Next.js
    route handler) or generates a UUID4 fallback. Binds trace_id, method, and
    path to structlog contextvars so all downstream log lines include them
    automatically. Returns the trace_id in the X-Request-ID response header.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        trace_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # bind_contextvars (not clear+bind) so parent-task context isn't clobbered.
        structlog.contextvars.bind_contextvars(
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)
        response.headers["X-Request-ID"] = trace_id
        return response


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Return a structured 504 if the request handler exceeds the timeout.

    Note: for SSE streaming routes, BaseHTTPMiddleware's call_next() returns
    as soon as response headers are sent — before the body streams. So this
    middleware only covers response-header latency on streaming routes; the
    actual stream timeout is handled inside _stream_chat() via asyncio.timeout.
    """

    def __init__(self, app: ASGIApp, timeout_seconds: float = 30.0) -> None:
        super().__init__(app)
        self.timeout = timeout_seconds

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Read trace_id from request header directly — avoids depending on
        # contextvars execution order relative to RequestIDMiddleware.
        trace_id = request.headers.get("X-Request-ID", "")
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            log.warning("api.timeout", timeout_seconds=self.timeout)
            body = ErrorResponse(error="Request timed out", trace_id=trace_id)
            return JSONResponse(status_code=504, content=body.model_dump())


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log status and latency for every non-health request.

    method and path come from structlog contextvars bound by RequestIDMiddleware.
    Health endpoints are excluded to avoid log noise from polling.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)

        if request.url.path not in _HEALTH_PATHS:
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            log.info("api.request", status=response.status_code, latency_ms=latency_ms)

        return response
