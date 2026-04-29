"""RetrieverTool — wraps :class:`~app.rag.retrieval.ensemble.EnsembleRetriever` for *agent tool* APIs.

This is **not** the LangGraph :mod:`app.orchestrator.langgraph.nodes.retriever` node. The graph node calls
``EnsembleRetriever`` directly inside the orchestration state machine. ``RetrieverTool`` exists
so a future chat/agent layer (ADK, LangChain tools, etc.) can expose the same retrieval as a
named tool (``search_knowledge_base``) with a Pydantic input schema — optional and unused until
you wire that surface.
"""

from __future__ import annotations

from pydantic import Field

from rag.retrieval.ensemble import EnsembleRetriever
from rag.tools.base import ToolInput, ToolOutput


class RetrieverToolInput(ToolInput):
    queries: list[str] = Field(min_length=1, max_length=3)
    num_results: int = Field(default=10, ge=1, le=50)


class RetrieverToolOutput(ToolOutput):
    results: list[dict]
    total: int


class RetrieverTool:
    """Multi-query retrieval tool backed by EnsembleRetriever."""

    name = "search_knowledge_base"
    description = "Multi-query hybrid search over the knowledge base"
    input_schema = RetrieverToolInput
    output_schema = RetrieverToolOutput

    def __init__(self, ensemble: EnsembleRetriever) -> None:
        self._ensemble = ensemble

    async def run(self, tool_input: RetrieverToolInput) -> RetrieverToolOutput:
        chunks = await self._ensemble.retrieve(
            tool_input.queries, k=tool_input.num_results
        )

        results = [
            {
                "text": gc.chunk.text,
                "url": gc.chunk.metadata.url,
                "title": gc.chunk.metadata.title,
                "score": round(gc.score, 4),
                "chunk_id": gc.chunk.id,
            }
            for gc in chunks
        ]

        return RetrieverToolOutput(results=results, total=len(results))


__all__ = ["RetrieverTool", "RetrieverToolInput", "RetrieverToolOutput"]
