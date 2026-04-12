from __future__ import annotations

import asyncio
import inspect
from typing import Any

from librarian.retrieval.base import Embedder, Retriever
from librarian.retrieval.cache import RetrievalCache
from librarian.schemas.chunks import GradedChunk
from librarian.schemas.retrieval import QueryPlan, RetrievalResult
from librarian.schemas.state import LibrarianState
from core.logging import get_logger

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


class RetrieverAgent:
    """Hybrid multi-query retrieval with relevance grading.

    Stateless agent: expands the query using plan.query_variants, embeds each
    variant, searches the vector store, deduplicates, and grades results by
    relevance threshold.
    """

    name = "retriever"
    description = "Hybrid multi-query retrieval with relevance grading"

    def __init__(
        self,
        retriever: Retriever,
        embedder: Embedder,
        top_k: int = 10,
        relevance_threshold: float = _RELEVANCE_THRESHOLD,
        cache: RetrievalCache | None = None,
        cache_strategy: str = "",
    ) -> None:
        self._retriever = retriever
        self._embedder = embedder
        self._top_k = top_k
        self._threshold = relevance_threshold
        self._cache = cache
        self._cache_strategy = cache_strategy

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

        cached_results: list[RetrievalResult] = []
        missed_variants: list[str] = []
        for variant in variants:
            cached = (
                self._cache.get(variant, self._cache_strategy, self._top_k)
                if self._cache is not None
                else None
            )
            if cached is None:
                missed_variants.append(variant)
                continue
            cached_results.extend(cached)

        async def _embed_variant(variant: str) -> list[float]:
            aembed_query = getattr(self._embedder, "aembed_query", None)
            if callable(aembed_query):
                result = aembed_query(variant)
                if inspect.isawaitable(result):
                    return await result
                return result  # sync aembed_query — use result directly
            return await asyncio.to_thread(self._embedder.embed_query, variant)

        all_results: list[RetrievalResult] = list(cached_results)
        if missed_variants:
            query_vectors = await asyncio.gather(
                *(_embed_variant(variant) for variant in missed_variants)
            )
            result_lists = await asyncio.gather(
                *(
                    self._retriever.search(
                        query_text=variant,
                        query_vector=query_vector,
                        k=self._top_k,
                    )
                    for variant, query_vector in zip(missed_variants, query_vectors)
                )
            )
            for variant, results in zip(missed_variants, result_lists):
                all_results.extend(results)
                if self._cache is not None:
                    self._cache.put(variant, self._cache_strategy, self._top_k, results)

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

    def as_node(self) -> Any:
        """Return a LangGraph-compatible async node function."""
        async def retrieve(state: LibrarianState) -> dict[str, Any]:
            result = await self.run(state)
            retry_count = int(state.get("retry_count") or 0)
            return {**result, "retry_count": retry_count}

        return retrieve
