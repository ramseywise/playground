"""RetrieverTool — wraps EnsembleRetriever as a framework-agnostic tool."""

from __future__ import annotations

from pydantic import Field

from librarian.retrieval.ensemble import EnsembleRetriever
from librarian.tools.base import ToolInput, ToolOutput


class RetrieverToolInput(ToolInput):
    """Input schema for the retriever tool."""

    queries: list[str] = Field(min_length=1, max_length=3)
    num_results: int = Field(default=10, ge=1, le=50)


class RetrieverToolOutput(ToolOutput):
    """Output schema for the retriever tool."""

    results: list[dict]
    total: int
    deduplicated: int


class RetrieverTool:
    """Multi-query retrieval tool backed by ``EnsembleRetriever``.

    Satisfies ``BaseTool[RetrieverToolInput, RetrieverToolOutput]``.
    """

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

        return RetrieverToolOutput(
            results=results,
            total=len(results),
            deduplicated=len(results),
        )
