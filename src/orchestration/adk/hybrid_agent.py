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
from core.logging import get_logger

log = get_logger(__name__)


def _extract_messages(events: list[Event]) -> list[dict[str, str]]:
    """Convert ADK session events into LangGraph-compatible message dicts."""
    messages: list[dict[str, str]] = []
    for event in events:
        if not event.content or not event.content.parts:
            continue
        text = ""
        for part in event.content.parts:
            if hasattr(part, "text") and part.text:
                text += part.text
        if not text:
            continue
        role = "user" if event.author == "user" else "assistant"
        messages.append({"role": role, "content": text})
    return messages


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
        query = _extract_latest_query(ctx)
        messages = _extract_messages(ctx.session.events)

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

        # Run the full pipeline
        result = await self._graph.ainvoke(
            state,
            config={"configurable": {"thread_id": ctx.session.id}},
        )

        response_text = result.get("response", "")
        citations = result.get("citations", [])

        log.info(
            "adk.hybrid.response",
            response_len=len(response_text),
            citation_count=len(citations) if isinstance(citations, list) else 0,
            confidence=result.get("confidence_score", 0.0),
        )

        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=response_text)],
            ),
        )


def _extract_latest_query(ctx: InvocationContext) -> str:
    """Extract the latest user message text from ADK session events."""
    for event in reversed(ctx.session.events):
        if event.author == "user" and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    return part.text
    return ""


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
