"""VA Assistant Gateway — FastAPI server with SSE streaming.

Endpoints:
  POST /chat                        Trigger a new agent turn (fires background task)
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
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

import shared.artefact_store as artefact_store
import shared.memory as memory_store

from .session_manager import _SENTINEL, session_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "4000"))


def _require_api_key(key: str | None = Security(_api_key_header)) -> None:
    if not _GATEWAY_API_KEY:
        return  # not configured — dev / local mode
    if key != _GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await memory_store.init_memory_db()
    await artefact_store.init_artefact_db()
    yield


app = FastAPI(title="VA Assistant Gateway", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/agents", dependencies=[Depends(_require_api_key)])
async def list_agents():
    return [
        {
            "name": "va_assistant",
            "description": "Billy accounting assistant — invoices, quotes, customers, products, emails, invitations, and support.",
        }
    ]


@app.post("/chat", dependencies=[Depends(_require_api_key)])
async def post_chat(
    req: ChatRequest,
    x_trace_id: str | None = Header(default=None),
):
    """Accept a user message and start an ADK turn as a background task.

    The client must open GET /chat/stream BEFORE calling this endpoint so that
    SSE events are not lost.
    """
    if len(req.message) > _MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=400, detail=f"Message exceeds {_MAX_MESSAGE_CHARS} characters")

    trace_id = x_trace_id or req.request_id

    # Ensure session + queue exist before the background task starts
    await session_manager.get_or_create(req.session_id, req.user_id)

    asyncio.create_task(
        session_manager.run_turn(
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
    """SSE stream for a session.

    Events:
      data: {"type": "text",     "data": "<markdown chunk>"}
      data: {"type": "response", "data": {<AssistantResponse>}}
      data: {"type": "error",    "data": "<error message>"}
      data: {"type": "done"}
    """
    session = await session_manager.get_or_create(session_id)

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
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "gateway.main:app",
        host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
        port=int(os.getenv("GATEWAY_PORT", "8000")),
        reload=True,
    )
