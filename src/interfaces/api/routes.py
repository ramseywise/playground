"""API routes for the Librarian RAG agent."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any, cast

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from interfaces.api.backends import BACKEND_LABELS
from interfaces.api.deps import (
    get_adk_bedrock_agent,
    get_adk_custom_rag_agent,
    get_adk_hybrid_agent,
    get_bedrock_client,
    get_google_adk_client,
    get_graph,
    get_pipeline,
    get_settings,
    get_triage,
    is_adk_available,
    is_bedrock_available,
    is_google_adk_available,
    is_graph_ready,
    verify_api_key,
)
from interfaces.api.models import (
    BackendInfo,
    BackendsResponse,
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


@router.get("/backends", response_model=BackendsResponse, dependencies=[Depends(verify_api_key)])
async def backends() -> BackendsResponse:
    """Return which backends are configured and available."""
    adk = is_adk_available()
    bedrock = is_bedrock_available()
    google = is_google_adk_available()
    graph = is_graph_ready()

    items = [
        BackendInfo(
            id="librarian",
            label=BACKEND_LABELS["librarian"],
            available=graph,
            streaming=True,
        ),
        BackendInfo(
            id="bedrock",
            label=BACKEND_LABELS["bedrock"],
            available=bedrock,
        ),
        BackendInfo(
            id="google_adk",
            label=BACKEND_LABELS["google_adk"],
            available=google,
        ),
        BackendInfo(
            id="adk_bedrock",
            label=BACKEND_LABELS["adk_bedrock"],
            available=adk and bedrock,
        ),
        BackendInfo(
            id="adk_custom_rag",
            label=BACKEND_LABELS["adk_custom_rag"],
            available=adk and graph,
        ),
        BackendInfo(
            id="adk_hybrid",
            label=BACKEND_LABELS["adk_hybrid"],
            available=adk and graph,
        ),
    ]
    return BackendsResponse(backends=items)


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(req: ChatRequest) -> ChatResponse | JSONResponse:
    """Non-streaming chat: triage → librarian graph, Bedrock KB, or direct response."""
    decision = get_triage().decide(req.query, req.backend)

    if decision.route == "bedrock":
        return await _chat_bedrock(req)
    if decision.route == "google_adk":
        return await _chat_google_adk(req)
    if decision.route == "adk_bedrock":
        return await _chat_adk_bedrock(req)
    if decision.route == "adk_custom_rag":
        return await _chat_adk_custom_rag(req)
    if decision.route == "adk_hybrid":
        return await _chat_adk_hybrid(req)
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

    thread_id = req.session_id or req.conversation_id or str(uuid.uuid4())
    handler = build_langfuse_handler(
        session_id=req.session_id or "",
        trace_id=trace_id,
    )
    config = make_runnable_config(handler, thread_id=thread_id)

    try:
        result: dict[str, Any] = await graph.ainvoke(
            {"query": req.query}, config=cast(Any, config)
        )
    except Exception:
        log.exception("api.chat.librarian.error")
        return _error_response(500, "Internal graph error")

    confident = result.get("confident", True)
    fallback_requested = result.get("fallback_requested", False)
    return ChatResponse(
        response=result.get("response", ""),
        citations=result.get("citations", []),
        confidence_score=result.get("confidence_score", 0.0),
        confident=confident,
        escalate=not confident or fallback_requested,
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


async def _chat_google_adk(req: ChatRequest) -> ChatResponse | JSONResponse:
    """Call Google Gemini with Vertex AI Search grounding."""
    client = get_google_adk_client()
    if client is None:
        return _error_response(
            503,
            "Google RAG not configured",
            detail="Set GOOGLE_DATASTORE_ID and GEMINI_API_KEY in .env",
        )

    log.info("api.chat.google_adk", query=req.query[:80], session_id=req.session_id)

    try:
        result = await client.aquery(req.query, session_id=req.session_id)
    except Exception:
        log.exception("api.chat.google_adk.error")
        return _error_response(502, "Google RAG error")

    return ChatResponse(
        response=result.response,
        citations=result.citations,
        confidence_score=0.0,
        intent="google_rag",
        trace_id=_trace_id(),
        backend="google_adk",
    )


async def _chat_adk_bedrock(req: ChatRequest) -> ChatResponse | JSONResponse:
    """Call Bedrock KB through the ADK agent wrapper."""
    agent = get_adk_bedrock_agent()
    if agent is None:
        return _error_response(
            503,
            "ADK Bedrock agent not available",
            detail="Requires google-adk and BEDROCK_KNOWLEDGE_BASE_ID in .env",
        )

    log.info("api.chat.adk_bedrock", query=req.query[:80], session_id=req.session_id)

    try:
        from orchestration.google_adk.context import build_adk_context, extract_urls_from_adk_events

        ctx, _ = build_adk_context(req.query, req.session_id or str(uuid.uuid4()))
        events = [e async for e in agent._run_async_impl(ctx)]

        response_text = ""
        if events and events[0].content and events[0].content.parts:
            response_text = events[0].content.parts[0].text or ""

        citations: list[dict[str, str]] = []
        for url in extract_urls_from_adk_events(events):
            citations.append({"url": url, "title": url.split("/")[-1]})

    except Exception:
        log.exception("api.chat.adk_bedrock.error")
        return _error_response(502, "ADK Bedrock error")

    return ChatResponse(
        response=response_text,
        citations=citations,
        confidence_score=0.0,
        intent="adk_bedrock",
        trace_id=_trace_id(),
        backend="adk_bedrock",
    )


async def _chat_adk_custom_rag(req: ChatRequest) -> ChatResponse | JSONResponse:
    """Call the ADK custom RAG agent (Gemini 2.0 Flash + tool-calling)."""
    agent = get_adk_custom_rag_agent()
    if agent is None:
        return _error_response(
            503,
            "ADK custom RAG agent not available",
            detail="Requires google-adk and a configured graph",
        )

    log.info("api.chat.adk_custom_rag", query=req.query[:80], session_id=req.session_id)

    try:
        from orchestration.google_adk.custom_rag_agent import run_custom_rag_query

        result = await run_custom_rag_query(
            agent,
            req.query,
            session_id=req.session_id or str(uuid.uuid4()),
        )

        response_text = result.get("response", "")
        citations: list[dict[str, str]] = []
        from orchestration.google_adk.context import extract_urls_from_adk_events

        for url in extract_urls_from_adk_events(result.get("events", [])):
            citations.append({"url": url, "title": url.split("/")[-1]})

    except Exception:
        log.exception("api.chat.adk_custom_rag.error")
        return _error_response(502, "ADK Custom RAG error")

    return ChatResponse(
        response=response_text,
        citations=citations,
        confidence_score=0.0,
        intent="adk_custom_rag",
        trace_id=_trace_id(),
        backend="adk_custom_rag",
    )


async def _chat_adk_hybrid(req: ChatRequest) -> ChatResponse | JSONResponse:
    """Call the ADK-wrapped LangGraph CRAG pipeline."""
    agent = get_adk_hybrid_agent()
    if agent is None:
        return _error_response(
            503,
            "ADK hybrid agent not available",
            detail="Requires google-adk and a configured graph",
        )

    log.info("api.chat.adk_hybrid", query=req.query[:80], session_id=req.session_id)

    try:
        from orchestration.google_adk.context import build_adk_context, extract_urls_from_adk_events

        ctx, _ = build_adk_context(req.query, req.session_id or str(uuid.uuid4()))
        events = [e async for e in agent._run_async_impl(ctx)]

        response_text = ""
        if events and events[0].content and events[0].content.parts:
            response_text = events[0].content.parts[0].text or ""

        citations: list[dict[str, str]] = []
        for url in extract_urls_from_adk_events(events):
            citations.append({"url": url, "title": url.split("/")[-1]})

    except Exception:
        log.exception("api.chat.adk_hybrid.error")
        return _error_response(502, "ADK Hybrid error")

    return ChatResponse(
        response=response_text,
        citations=citations,
        confidence_score=0.0,
        intent="adk_hybrid",
        trace_id=_trace_id(),
        backend="adk_hybrid",
    )


async def _stream_chat(
    req: ChatRequest, timeout: float
) -> AsyncGenerator[dict[str, Any], None]:
    """Run the graph with astream(), emitting SSE events per node."""
    graph = get_graph()
    trace_id = _trace_id()

    log.info("api.chat.stream", query=req.query[:80], session_id=req.session_id)

    thread_id = req.session_id or req.conversation_id or str(uuid.uuid4())
    handler = build_langfuse_handler(
        session_id=req.session_id or "",
        trace_id=trace_id,
    )
    config = make_runnable_config(handler, thread_id=thread_id)

    # Only extract the fields we need — avoids holding large state (e.g.
    # raw chunks or embeddings) in memory for the full stream duration.
    response = citations = confidence = intent = None
    confident: bool = True
    fallback_requested: bool = False

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
                        if "confident" in update:
                            confident = update["confident"]
                        if "fallback_requested" in update:
                            fallback_requested = update["fallback_requested"]
                    yield {"event": "status", "data": {"stage": node_name}}

        escalate = not confident or fallback_requested
        yield {
            "event": "done",
            "data": {
                "response": response or "",
                "citations": citations or [],
                "confidence_score": confidence or 0.0,
                "confident": confident,
                "escalate": escalate,
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


@router.post("/chat/stream", dependencies=[Depends(verify_api_key)])
async def chat_stream(req: ChatRequest) -> EventSourceResponse:
    """Streaming chat: triage first, then emit SSE events."""
    decision = get_triage().decide(req.query, req.backend)

    if decision.route in ("escalation", "direct"):

        async def triage_events() -> AsyncGenerator[str, None]:
            yield format_sse("status", {"stage": "triage"})
            yield format_sse(
                "done",
                {
                    "response": decision.response or "",
                    "citations": [],
                    "confidence_score": decision.confidence,
                    "intent": decision.intent,
                    "trace_id": _trace_id(),
                },
            )

        return EventSourceResponse(triage_events())

    if decision.route in (
        "bedrock",
        "google_adk",
        "adk_bedrock",
        "adk_custom_rag",
        "adk_hybrid",
    ):
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"Streaming is not supported for the {decision.route} backend. Use /chat instead.",
            },
        )

    cfg = get_settings()

    async def event_generator() -> AsyncGenerator[str, None]:
        async for evt in _stream_chat(req, timeout=cfg.api_stream_timeout_seconds):
            yield format_sse(evt["event"], evt["data"])

    return EventSourceResponse(event_generator())


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
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
            r = await pipeline.ingest_document(req.document.model_dump())
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
