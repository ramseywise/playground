"""HTTP API entrypoint for container / uvicorn."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import orjson
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from core.config import (
    ALLOWED_ORIGINS,
    CHECKPOINTER_BACKEND,
    DATABASE_URL,
    RAG_GUARDRAILS_ENABLED,
    SQLITE_PATH,
)
from core.observability import configure_runtime
from orchestrator.runtime_protocol import AgentRuntime
from orchestrator.schemas import AgentInput, AgentOutput, ResumeInput, StreamEvent

log = logging.getLogger(__name__)

_runtime: AgentRuntime | None = None


# ---------------------------------------------------------------------------
# Runtime builder
# ---------------------------------------------------------------------------


async def _build_langgraph_runtime() -> tuple[AgentRuntime, object | None]:
    """Build LangGraphRuntime with the configured checkpointer."""
    from orchestrator.langgraph.graph import build_graph
    from orchestrator.langgraph.runtime import LangGraphRuntime

    if CHECKPOINTER_BACKEND == "postgres":
        if not DATABASE_URL:
            raise RuntimeError(
                "CHECKPOINTER_BACKEND=postgres but DATABASE_URL is not set. "
                "Add DATABASE_URL=postgresql+psycopg://user:pass@host/db to your environment."
            )
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as exc:
            raise RuntimeError(
                "langgraph-checkpoint-postgres is not installed. "
                "Run: uv add hc_agent_rag[postgres]"
            ) from exc

        ctx = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
        cp = await ctx.__aenter__()
        await cp.setup()
        log.info("startup: checkpointer=postgres")
        return LangGraphRuntime(graph=build_graph(cp)), ctx

    if CHECKPOINTER_BACKEND == "sqlite":
        import os
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # type: ignore[import-untyped]

        os.makedirs(os.path.dirname(os.path.abspath(SQLITE_PATH)), exist_ok=True)
        ctx = AsyncSqliteSaver.from_conn_string(SQLITE_PATH)
        cp = await ctx.__aenter__()
        log.info("startup: checkpointer=sqlite path=%s", SQLITE_PATH)
        return LangGraphRuntime(graph=build_graph(cp)), ctx

    from langgraph.checkpoint.memory import MemorySaver

    log.info("startup: checkpointer=memory (sessions not persisted)")
    return LangGraphRuntime(graph=build_graph(MemorySaver())), None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _runtime
    configure_runtime()

    _cp_ctx: object | None = None

    log.info("startup: backend=langgraph checkpointer=%s", CHECKPOINTER_BACKEND)
    _runtime, _cp_ctx = await _build_langgraph_runtime()

    if RAG_GUARDRAILS_ENABLED:
        log.info("startup: guardrails=enabled")
    log.info("startup: ready")
    yield

    if _cp_ctx is not None:
        await _cp_ctx.__aexit__(None, None, None)  # type: ignore[attr-defined]
    log.info("shutdown: complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


app = FastAPI(
    title="hc-rag-agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Session-Id"],
    allow_credentials=True,
)


# ---------------------------------------------------------------------------
# Guardrail middleware
# ---------------------------------------------------------------------------


if RAG_GUARDRAILS_ENABLED:
    from starlette.middleware.base import BaseHTTPMiddleware

    class _GuardrailMiddleware(BaseHTTPMiddleware):
        """Reject PII-laden or injection-attempt requests before they reach the runtime."""

        _CHAT_PATHS = frozenset(["/api/v1/chat", "/api/v1/chat/stream"])

        async def dispatch(self, request: Request, call_next):  # type: ignore[override, no-untyped-def]
            if request.method == "POST" and request.url.path in self._CHAT_PATHS:
                try:
                    body = await request.body()
                    data = orjson.loads(body)
                    query: str = data.get("query", "") or ""
                except Exception:
                    query = ""

                if query:
                    from guardrails import detect_and_redact, looks_like_injection

                    redacted, pii_found = detect_and_redact(query)
                    if pii_found:
                        log.warning("guardrails: pii_detected path=%s", request.url.path)

                    if looks_like_injection(redacted):
                        log.warning("guardrails: injection_blocked path=%s", request.url.path)
                        return Response(
                            content=orjson.dumps({"detail": "Request blocked by guardrails."}),
                            status_code=400,
                            media_type="application/json",
                        )

            return await call_next(request)

    app.add_middleware(_GuardrailMiddleware)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runtime_or_503() -> AgentRuntime:
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Runtime not ready")
    return _runtime


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@app.post("/api/v1/sessions", status_code=201)
def create_session() -> dict[str, str]:
    """Mint a new session ID for a conversation thread."""
    return {"session_id": str(uuid.uuid4())}


# ---------------------------------------------------------------------------
# Non-streaming chat
# ---------------------------------------------------------------------------


@app.post("/api/v1/chat", response_model=AgentOutput)
async def chat(body: AgentInput) -> AgentOutput:
    rt = _runtime_or_503()
    if not body.thread_id:
        raise HTTPException(status_code=422, detail="thread_id is required")
    t0 = time.perf_counter()
    log.info("POST /api/v1/chat thread_id=%s", body.thread_id)
    result = await rt.run(body)
    log.info("POST /api/v1/chat done thread_id=%s elapsed_ms=%.1f", body.thread_id, (time.perf_counter() - t0) * 1000)
    return result


# ---------------------------------------------------------------------------
# SSE streaming chat
# ---------------------------------------------------------------------------


def _sse(event: StreamEvent) -> str:
    """Encode a StreamEvent as a single SSE frame."""
    return f"event: {event.kind}\ndata: {orjson.dumps(event.data).decode()}\n\n"


async def _stream_events(rt: AgentRuntime, body: AgentInput) -> AsyncIterator[str]:
    t_first: float | None = None
    t0 = time.perf_counter()
    async for ev in rt.stream(body):
        if t_first is None and ev.kind in ("token", "node_start"):
            t_first = (time.perf_counter() - t0) * 1000
        yield _sse(ev)
    log.info(
        "stream done thread_id=%s first_event_ms=%s total_ms=%.1f",
        body.thread_id,
        f"{t_first:.1f}" if t_first is not None else "n/a",
        (time.perf_counter() - t0) * 1000,
    )


@app.post("/api/v1/chat/stream")
async def chat_stream(body: AgentInput, request: Request) -> StreamingResponse:
    rt = _runtime_or_503()
    if not body.thread_id:
        raise HTTPException(status_code=422, detail="thread_id is required")
    log.info("POST /api/v1/chat/stream thread_id=%s", body.thread_id)
    return StreamingResponse(
        _stream_events(rt, body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# HITL resume (LangGraph only)
# ---------------------------------------------------------------------------


@app.post("/api/v1/chat/resume", response_model=AgentOutput)
async def chat_resume(body: ResumeInput) -> AgentOutput:
    """Resume a graph paused at a HITL interrupt."""
    rt = _runtime_or_503()
    log.info("POST /api/v1/chat/resume thread_id=%s", body.thread_id)
    return await rt.resume(body.thread_id, body.value)


@app.post("/api/v1/chat/resume/stream")
async def chat_resume_stream(body: ResumeInput, request: Request) -> StreamingResponse:
    """Resume a HITL interrupt and stream remaining events."""
    rt = _runtime_or_503()
    log.info("POST /api/v1/chat/resume/stream thread_id=%s", body.thread_id)

    async def _gen() -> AsyncIterator[str]:
        async for ev in rt.stream_resume(body.thread_id, body.value):
            yield _sse(ev)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> Response:
    return Response(
        content=orjson.dumps({"status": "ok", "backend": "langgraph"}),
        media_type="application/json",
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}
