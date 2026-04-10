from __future__ import annotations

from typing import Any

from agents.librarian.retrieval.base import Embedder, Retriever
from agents.librarian.schemas.chunks import GradedChunk
from agents.librarian.schemas.retrieval import QueryPlan, RetrievalResult
from agents.librarian.schemas.state import LibrarianState
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)

# Chunk is relevant when its hybrid score clears this threshold.
_RELEVANCE_THRESHOLD = 0.1


def _grade_chunks(
    results: list[RetrievalResult],
    threshold: float = _RELEVANCE_THRESHOLD,
) -> list[GradedChunk]:
    """Convert raw retrieval results to GradedChunks, deduplicating by chunk ID."""
    seen: set[str] = set()
    graded: list[GradedChunk] = []
    for r in results:
        if r.chunk.id in seen:
            continue
        seen.add(r.chunk.id)
        graded.append(
            GradedChunk(
                chunk=r.chunk,
                score=r.score,
                relevant=r.score >= threshold,
            )
        )
    return graded


class RetrievalSubgraph:
    """Stateless node: retrieve → deduplicate → grade.

    Expands the query using plan.query_variants (multi-query).
    Falls back to state["query"] / state["standalone_query"] when no plan.
    """

    def __init__(
        self,
        retriever: Retriever,
        embedder: Embedder,
        top_k: int = 10,
        relevance_threshold: float = _RELEVANCE_THRESHOLD,
    ) -> None:
        self._retriever = retriever
        self._embedder = embedder
        self._top_k = top_k
        self._threshold = relevance_threshold

    async def run(self, state: LibrarianState) -> dict[str, Any]:
        plan: QueryPlan | None = state.get("plan")
        base_query = state.get("standalone_query") or state.get("query", "")

        variants: list[str] = (
            list(plan.query_variants) if plan and plan.query_variants else []
        )
        if not variants:
            variants = [base_query]
        elif base_query and base_query not in variants:
            variants.insert(0, base_query)

        log.info(
            "retrieval.subgraph.start",
            base_query=base_query,
            variant_count=len(variants),
            top_k=self._top_k,
        )

        all_results: list[RetrievalResult] = []
        for variant in variants:
            query_vector = self._embedder.embed_query(variant)
            results = await self._retriever.search(
                query_text=variant,
                query_vector=query_vector,
                k=self._top_k,
            )
            all_results.extend(results)

        graded = _grade_chunks(all_results, threshold=self._threshold)

        log.info(
            "retrieval.subgraph.done",
            raw_results=len(all_results),
            unique_chunks=len(graded),
            relevant=sum(1 for g in graded if g.relevant),
        )

        return {
            "retrieved_chunks": all_results,
            "graded_chunks": graded,
            "query_variants": variants,
        }
