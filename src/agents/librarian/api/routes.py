"""API routes for the Librarian RAG agent."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from agents.librarian.api.deps import get_generation_subgraph, get_graph, get_pipeline, get_settings
from agents.librarian.api.models import ChatRequest, ChatResponse, IngestRequest, IngestResponse, IngestResultItem
from agents.librarian.api.streaming import format_sse
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Non-streaming chat: run the full graph and return the result."""
    graph = get_graph()

    log.info("api.chat", query=req.query[:80])

    try:
        result: dict[str, Any] = await graph.ainvoke({"query": req.query})
    except Exception:
        log.exception("api.chat.error")
        raise HTTPException(status_code=500, detail="Internal graph error")

    return ChatResponse(
        response=result.get("response", ""),
        citations=result.get("citations", []),
        confidence_score=result.get("confidence_score", 0.0),
        intent=result.get("intent", ""),
    )


async def _stream_chat(req: ChatRequest) -> AsyncGenerator[dict[str, Any], None]:
    """Run the graph with astream(), emitting SSE events per node."""
    graph = get_graph()
    gen_sg = get_generation_subgraph()

    log.info("api.chat.stream", query=req.query[:80])

    # Collect final state as we stream through graph nodes
    final_state: dict[str, Any] = {"query": req.query}

    try:
        async for event in graph.astream({"query": req.query}):
            # astream yields {node_name: state_update} dicts
            for node_name, update in event.items():
                if isinstance(update, dict):
                    final_state.update(update)
                yield {"event": "status", "data": {"stage": node_name}}

        # After graph completes, stream the generation tokens if we have
        # reranked chunks and want token-level streaming
        # For MVP: the graph already ran generation, so emit the final response
        response = final_state.get("response", "")
        citations = final_state.get("citations", [])
        confidence = final_state.get("confidence_score", 0.0)
        intent = final_state.get("intent", "")

        yield {
            "event": "done",
            "data": {
                "response": response,
                "citations": citations,
                "confidence_score": confidence,
                "intent": intent,
            },
        }
    except Exception:
        log.exception("api.chat.stream.error")
        yield {"event": "error", "data": {"detail": "Internal graph error"}}


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> EventSourceResponse:
    """Streaming chat: emit SSE events as the graph progresses."""

    async def event_generator() -> AsyncGenerator[str, None]:
        async for evt in _stream_chat(req):
            yield format_sse(evt["event"], evt["data"])

    return EventSourceResponse(event_generator())


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    """Ingest documents from S3 or inline payload."""
    pipeline = get_pipeline()
    cfg = get_settings()

    log.info("api.ingest", s3_key=req.s3_key, s3_prefix=req.s3_prefix, inline=req.document is not None)

    results: list[IngestResultItem] = []

    try:
        if req.s3_key:
            r = await pipeline.ingest_s3_object(
                bucket=cfg.s3_bucket, key=req.s3_key, region=cfg.s3_region,
            )
            results.append(IngestResultItem(
                doc_id=r.doc_id, chunk_count=r.chunk_count,
                snippet_count=r.snippet_count, skipped=r.skipped,
            ))

        if req.s3_prefix:
            batch = await pipeline.ingest_s3_prefix(
                bucket=cfg.s3_bucket, prefix=req.s3_prefix, region=cfg.s3_region,
            )
            for r in batch:
                results.append(IngestResultItem(
                    doc_id=r.doc_id, chunk_count=r.chunk_count,
                    snippet_count=r.snippet_count, skipped=r.skipped,
                ))

        if req.document:
            r = await pipeline.ingest_document(req.document)
            results.append(IngestResultItem(
                doc_id=r.doc_id, chunk_count=r.chunk_count,
                snippet_count=r.snippet_count, skipped=r.skipped,
            ))
    except Exception:
        log.exception("api.ingest.error")
        raise HTTPException(status_code=500, detail="Ingestion error")

    log.info("api.ingest.done", result_count=len(results))
    return IngestResponse(results=results)
