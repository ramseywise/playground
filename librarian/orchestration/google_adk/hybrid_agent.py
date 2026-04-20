"""Option 4: ADK BaseAgent wrapping the full LangGraph Librarian pipeline.

Combines ADK's session management and multi-agent routing with the
Librarian's full CRAG pipeline (condense → analyze → retrieve → rerank →
gate → generate). This tests whether ADK adds value as an outer shell
around an already-capable pipeline.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from langgraph.graph.state import CompiledStateGraph

from librarian.config import LibrarySettings
from orchestration.google_adk.utils import extract_latest_query, extract_messages
from core.logging import get_logger

log = get_logger(__name__)


class LibrarianADKAgent(BaseAgent):
    """ADK wrapper around the compiled LangGraph Librarian pipeline.

    Exposes the full CRAG pipeline (condense → analyze → retrieve →
    rerank → gate → generate) as a single ADK-compatible agent.

    The LangGraph pipeline handles all retrieval decisions internally —
    this wrapper just bridges ADK's session/event model to LangGraph's
    state dict model.
    """

    _graph: CompiledStateGraph
    _cfg: LibrarySettings | None

    def __init__(
        self,
        graph: CompiledStateGraph,
        cfg: LibrarySettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="librarian_hybrid",
            description=(
                "Full CRAG pipeline via LangGraph, wrapped as ADK agent. "
                "Condense → analyze → retrieve → rerank → gate → generate."
            ),
            **kwargs,
        )
        object.__setattr__(self, "_graph", graph)
        object.__setattr__(self, "_cfg", cfg)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Extract query from ADK session, run full LangGraph pipeline, emit response."""
        query = extract_latest_query(ctx)
        messages = extract_messages(ctx.session.events)

        log.info(
            "adk.hybrid.query",
            query=query[:80],
            session_id=ctx.session.id,
            message_count=len(messages),
        )

        # Build LangGraph state from ADK context
        state: dict[str, Any] = {
            "query": query,
            "messages": messages,
        }

        # Build LangGraph config with Langfuse tracing when available
        from librarian.tracing import build_langfuse_handler, make_runnable_config

        handler = build_langfuse_handler(
            session_id=ctx.session.id,
            trace_id=f"adk-hybrid-{ctx.session.id}",
        )
        config = make_runnable_config(handler, thread_id=ctx.session.id)

        # Run the full pipeline with tracing
        result = await self._graph.ainvoke(state, config=config)

        response_text = result.get("response", "")
        citations = result.get("citations", [])
        confidence = result.get("confidence_score", 0.0)

        log.info(
            "adk.hybrid.response",
            response_len=len(response_text),
            citation_count=len(citations) if isinstance(citations, list) else 0,
            confidence=confidence,
        )

        # Surface full pipeline metadata for debugging and eval
        reranked = result.get("reranked_chunks", [])
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=response_text)],
            ),
            custom_metadata={
                "citations": citations,
                "confidence_score": confidence,
                "intent": result.get("intent", ""),
                "standalone_query": result.get("standalone_query", ""),
                "retry_count": result.get("retry_count", 0),
                "reranked_count": len(reranked),
                "retrieved_urls": [
                    r.chunk.metadata.url for r in reranked if hasattr(r, "chunk")
                ][:10],
            },
        )


def create_hybrid_agent(
    cfg: LibrarySettings | None = None,
) -> LibrarianADKAgent:
    """Build a LibrarianADKAgent using the standard factory.

    This creates the full LangGraph pipeline via ``create_librarian()``
    and wraps it in an ADK BaseAgent.
    """
    from orchestration.factory import create_librarian

    graph = create_librarian(cfg)

    log.info(
        "adk.hybrid.created",
        retrieval_strategy=cfg.retrieval_strategy if cfg else "default",
        reranker_strategy=cfg.reranker_strategy if cfg else "default",
    )

    return LibrarianADKAgent(graph=graph, cfg=cfg)
