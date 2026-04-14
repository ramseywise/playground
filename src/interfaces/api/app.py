"""FastAPI application for the Librarian RAG agent."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from interfaces.api.deps import get_settings, init_graph, init_pipeline, init_triage
from interfaces.api.middleware import (
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    TimeoutMiddleware,
)
from interfaces.api.routes import router
from librarian.config import LibrarySettings
from core.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise the graph on startup."""
    settings = get_settings()
    configure_logging(render_json=settings.log_json)
    log.info("api.startup")
    init_graph()
    init_pipeline()
    init_triage()
    yield
    log.info("api.shutdown")


app = FastAPI(
    title="Librarian RAG Agent",
    description="Multi-source RAG agent with hybrid retrieval, reranking, and CRAG.",
    version="0.1.0",
    lifespan=lifespan,
)

cfg = get_settings()

# Middleware order matters — outermost runs first on request, last on response.
# RequestID must be first so trace_id is bound before logging/timeout see it.
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(TimeoutMiddleware, timeout_seconds=cfg.api_timeout_seconds)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


# Also mount health at root for ALB/ECS health checks
@app.get("/health")
async def root_health() -> dict[str, str]:
    return {"status": "ok"}


def main() -> None:
    """Entry point for ``librarian-api`` console script."""
    settings = get_settings()
    uvicorn.run(
        "interfaces.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
