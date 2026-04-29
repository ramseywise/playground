"""VA LangGraph Gateway — FastAPI server with SSE streaming.

Identical API contract to va-google-adk/gateway/main.py so the same
web client and tests work against both backends.

Endpoints:
  POST /chat                        Trigger a new graph turn (fires background task)
  GET  /chat/stream                 SSE stream for a session
  GET  /agents                      List available agents
  GET  /health                      Health check
  POST /artefacts                   Store generated content, returns {artefact_id, url}
  GET  /artefacts/{id}/download     Stream or redirect to artefact file
  DELETE /artefacts/{id}            Soft-delete an artefact
"""

from __future__ import annotations

import asyncio
import json
import logging  # noqa: use-structlog — configures LangChain stdlib output via logging.basicConfig
import os
from contextlib import asynccontextmanager

import structlog
from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

import artefact_store as artefact_store
import memory as memory_store

from .runner import _SENTINEL, runner

logging.basicConfig(
    level=logging.INFO
)  # configure stdlib for LangChain/LangGraph output
log = structlog.get_logger(__name__)

_CHECKPOINTER_BACKEND = os.getenv("LANGGRAPH_CHECKPOINTER", "memory")
_POSTGRES_URL = os.getenv("POSTGRES_URL", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await memory_store.init_memory_db()
    await artefact_store.init_artefact_db()

    if _CHECKPOINTER_BACKEND == "postgres" and _POSTGRES_URL:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        async with AsyncConnectionPool(conninfo=_POSTGRES_URL, max_size=10) as pool:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            runner.init_graph(checkpointer)
            log.info("checkpointer.ready", backend="postgres")
            yield
    else:
        from langgraph.checkpoint.memory import MemorySaver

        runner.init_graph(MemorySaver())
        log.info("checkpointer.ready", backend="memory")
        yield


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(key: str | None = Security(_api_key_header)) -> None:
    if not _GATEWAY_API_KEY:
        return  # not configured — dev / local mode
    if key != _GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


app = FastAPI(title="VA LangGraph Gateway", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    request_id: str
    message: str
    page_url: str | None = None
    user_id: str = "default"


class ArtefactRequest(BaseModel):
    session_id: str
    content: str
    filename: str
    content_type: str = "text/markdown"
    ttl_days: int = 30


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/agents", dependencies=[Depends(_require_api_key)])
async def list_agents():
    return [
        {
            "name": "va_assistant",
            "description": "Billy accounting assistant (LangGraph) — invoices, quotes, customers, products, emails, invitations, and support.",
        }
    ]


@app.post("/chat", dependencies=[Depends(_require_api_key)])
async def post_chat(
    req: ChatRequest,
    x_trace_id: str | None = Header(default=None),
):
    trace_id = x_trace_id or req.request_id
    runner.get_or_create(req.session_id)

    asyncio.create_task(
        runner.run_turn(
            session_id=req.session_id,
            message=req.message,
            page_url=req.page_url,
            trace_id=trace_id,
            user_id=req.user_id,
        )
    )
    return {"status": "accepted", "request_id": req.request_id, "trace_id": trace_id}


@app.get("/chat/stream", dependencies=[Depends(_require_api_key)])
async def stream_chat(session_id: str = Query(...)):
    """SSE stream — same event types as va-google-adk gateway."""
    session = runner.get_or_create(session_id)

    async def event_generator():
        while True:
            item = await session.queue.get()
            if item is _SENTINEL:
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Artefact endpoints
# ---------------------------------------------------------------------------


@app.post("/artefacts", dependencies=[Depends(_require_api_key)])
async def create_artefact(req: ArtefactRequest):
    """Store generated content and return {artefact_id, url}."""
    result = await artefact_store.save(
        session_id=req.session_id,
        content=req.content,
        filename=req.filename,
        content_type=req.content_type,
        ttl_days=req.ttl_days,
    )
    return result


@app.get("/artefacts/{artefact_id}/download")
async def download_artefact(artefact_id: str):
    """Stream the artefact file (local backend) or raise 404."""
    result = await artefact_store.read_local(artefact_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Artefact not found")
    data, content_type = result
    return Response(content=data, media_type=content_type)


@app.delete("/artefacts/{artefact_id}", dependencies=[Depends(_require_api_key)])
async def delete_artefact(artefact_id: str):
    """Soft-delete an artefact."""
    record = await artefact_store.get(artefact_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Artefact not found")
    await artefact_store.soft_delete(artefact_id)
    return {"status": "deleted", "artefact_id": artefact_id}


if __name__ == "__main__":
    uvicorn.run(
        "gateway.main:app",
        host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
        port=int(os.getenv("GATEWAY_PORT", "8001")),
        reload=True,
    )
