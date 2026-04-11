"""API routes for the Librarian RAG agent."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, cast

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from interfaces.api.deps import (
    get_bedrock_client,
    get_graph,
    get_pipeline,
    get_settings,
    get_triage,
)
from interfaces.api.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    IngestRequest,
    IngestResponse,
    IngestResultItem,
)
from interfaces.api.streaming import format_sse
from librarian.tracing import build_langfuse_handler, make_runnable_config
from core.logging import get_logger

log = get_logger(__name__)
router = APIRouter()


def _trace_id() -> str:
    return structlog.contextvars.get_contextvars().get("trace_id", "")


def _error_response(status: int, message: str, detail: str = "") -> JSONResponse:
    body = ErrorResponse(error=message, trace_id=_trace_id(), detail=detail)
    return JSONResponse(status_code=status, content=body.model_dump())


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse | JSONResponse:
    """Non-streaming chat: triage → librarian graph, Bedrock KB, or direct response."""
    decision = get_triage().decide(req.query, req.backend)

    if decision.route == "bedrock":
        return await _chat_bedrock(req)
    if decision.route in ("escalation", "direct"):
        return ChatResponse(
            response=decision.response or "",
            citations=[],
            confidence_score=decision.confidence,
            intent=decision.intent,
            trace_id=_trace_id(),
            backend="triage",
        )
    return await _chat_librarian(req)


async def _chat_librarian(req: ChatRequest) -> ChatResponse | JSONResponse:
    """Run the full librarian graph and return the result."""
    graph = get_graph()
    trace_id = _trace_id()

    log.info("api.chat.librarian", query=req.query[:80], session_id=req.session_id)

    handler = build_langfuse_handler(
        session_id=req.session_id or "",
        trace_id=trace_id,
    )
    config = make_runnable_config(handler)

    try:
        result: dict[str, Any] = await graph.ainvoke(
            {"query": req.query}, config=cast(Any, config)
        )
    except Exception:
        log.exception("api.chat.librarian.error")
        return _error_response(500, "Internal graph error")

    return ChatResponse(
        response=result.get("response", ""),
        citations=result.get("citations", []),
        confidence_score=result.get("confidence_score", 0.0),
        intent=result.get("intent", ""),
        trace_id=_trace_id(),
        backend="librarian",
    )


async def _chat_bedrock(req: ChatRequest) -> ChatResponse | JSONResponse:
    """Call Bedrock Knowledge Bases RetrieveAndGenerate."""
    client = get_bedrock_client()
    if client is None:
        return _error_response(
            503,
            "Bedrock KB not configured",
            detail="Set BEDROCK_KNOWLEDGE_BASE_ID and BEDROCK_MODEL_ARN in .env",
        )

    log.info("api.chat.bedrock", query=req.query[:80], session_id=req.session_id)

    try:
        result = await client.aquery(req.query, session_id=req.session_id)
    except Exception:
        log.exception("api.chat.bedrock.error")
        return _error_response(502, "Bedrock KB error")

    return ChatResponse(
        response=result.response,
        citations=result.citations,
        confidence_score=0.0,
        intent="bedrock_kb",
        trace_id=_trace_id(),
        backend="bedrock",
    )


async def _stream_chat(
    req: ChatRequest, timeout: float
) -> AsyncGenerator[dict[str, Any], None]:
    """Run the graph with astream(), emitting SSE events per node."""
    graph = get_graph()
    trace_id = _trace_id()

    log.info("api.chat.stream", query=req.query[:80], session_id=req.session_id)

    handler = build_langfuse_handler(
        session_id=req.session_id or "",
        trace_id=trace_id,
    )
    config = make_runnable_config(handler)

    # Only extract the four fields we need — avoids holding large state (e.g.
    # raw chunks or embeddings) in memory for the full stream duration.
    response = citations = confidence = intent = None

    try:
        async with asyncio.timeout(timeout):
            async for event in graph.astream(
                {"query": req.query}, config=cast(Any, config)
            ):
                for node_name, update in event.items():
                    if isinstance(update, dict):
                        if "response" in update:
                            response = update["response"]
                        if "citations" in update:
                            citations = update["citations"]
                        if "confidence_score" in update:
                            confidence = update["confidence_score"]
                        if "intent" in update:
                            intent = update["intent"]
                    yield {"event": "status", "data": {"stage": node_name}}

        yield {
            "event": "done",
            "data": {
                "response": response or "",
                "citations": citations or [],
                "confidence_score": confidence or 0.0,
                "intent": intent or "",
                "trace_id": trace_id,
            },
        }
    except asyncio.TimeoutError:
        log.warning("api.chat.stream.timeout")
        yield {
            "event": "error",
            "data": {"detail": "Graph timed out", "trace_id": trace_id},
        }
    except Exception:
        log.exception("api.chat.stream.error")
        yield {
            "event": "error",
            "data": {"detail": "Internal graph error", "trace_id": trace_id},
        }


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> EventSourceResponse:
    """Streaming chat: triage first, then emit SSE events."""
    decision = get_triage().decide(req.query, req.backend)

    if decision.route in ("escalation", "direct"):

        async def triage_events() -> AsyncGenerator[str, None]:
            yield format_sse("status", {"stage": "triage"})
            yield format_sse("done", {
                "response": decision.response or "",
                "citations": [],
                "confidence_score": decision.confidence,
                "intent": decision.intent,
                "trace_id": _trace_id(),
            })

        return EventSourceResponse(triage_events())

    cfg = get_settings()

    async def event_generator() -> AsyncGenerator[str, None]:
        async for evt in _stream_chat(req, timeout=cfg.api_stream_timeout_seconds):
            yield format_sse(evt["event"], evt["data"])

    return EventSourceResponse(event_generator())


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse | JSONResponse:
    """Ingest documents from S3 or inline payload."""
    pipeline = get_pipeline()
    cfg = get_settings()

    log.info(
        "api.ingest",
        s3_key=req.s3_key,
        s3_prefix=req.s3_prefix,
        inline=req.document is not None,
    )

    results: list[IngestResultItem] = []

    try:
        if req.s3_key:
            r = await pipeline.ingest_s3_object(
                bucket=cfg.s3_bucket,
                key=req.s3_key,
                region=cfg.s3_region,
            )
            results.append(
                IngestResultItem(
                    doc_id=r.doc_id,
                    chunk_count=r.chunk_count,
                    snippet_count=r.snippet_count,
                    skipped=r.skipped,
                )
            )

        if req.s3_prefix:
            batch = await pipeline.ingest_s3_prefix(
                bucket=cfg.s3_bucket,
                prefix=req.s3_prefix,
                region=cfg.s3_region,
            )
            for r in batch:
                results.append(
                    IngestResultItem(
                        doc_id=r.doc_id,
                        chunk_count=r.chunk_count,
                        snippet_count=r.snippet_count,
                        skipped=r.skipped,
                    )
                )

        if req.document:
            r = await pipeline.ingest_document(req.document)
            results.append(
                IngestResultItem(
                    doc_id=r.doc_id,
                    chunk_count=r.chunk_count,
                    snippet_count=r.snippet_count,
                    skipped=r.skipped,
                )
            )
    except Exception:
        log.exception("api.ingest.error")
        return _error_response(500, "Ingestion error")

    log.info("api.ingest.done", result_count=len(results))
    return IngestResponse(results=results)
