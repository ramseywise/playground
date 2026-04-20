"""VA Assistant Gateway — FastAPI server with SSE streaming.

Endpoints:
  POST /chat             Trigger a new agent turn (fires background task)
  GET  /chat/stream      SSE stream for a session
  GET  /agents           List available agents
  GET  /health           Health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .session_manager import _SENTINEL, session_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VA Assistant Gateway")

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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/agents")
async def list_agents():
    return [
        {
            "name": "va_assistant",
            "description": "Billy accounting assistant — invoices, quotes, customers, products, emails, invitations, and support.",
        }
    ]


@app.post("/chat")
async def post_chat(req: ChatRequest):
    """Accept a user message and start an ADK turn as a background task.

    The client must open GET /chat/stream BEFORE calling this endpoint so that
    SSE events are not lost.
    """
    # Ensure session + queue exist before the background task starts
    session_manager.get_or_create(req.session_id)

    asyncio.create_task(
        session_manager.run_turn(
            session_id=req.session_id,
            message=req.message,
            page_url=req.page_url,
        )
    )
    return {"status": "accepted", "request_id": req.request_id}


@app.get("/chat/stream")
async def stream_chat(session_id: str = Query(...)):
    """SSE stream for a session.

    Events:
      data: {"type": "text",     "data": "<markdown chunk>"}
      data: {"type": "response", "data": {<AssistantResponse>}}
      data: {"type": "error",    "data": "<error message>"}
      data: {"type": "done"}
    """
    session = session_manager.get_or_create(session_id)

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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "gateway.main:app",
        host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
        port=int(os.getenv("GATEWAY_PORT", "8000")),
        reload=True,
    )
