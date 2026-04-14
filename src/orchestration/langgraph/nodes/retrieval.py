from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from librarian.retrieval.base import Embedder, Retriever
from librarian.retrieval.cache import RetrievalCache
from librarian.schemas.chunks import GradedChunk
from librarian.schemas.retrieval import QueryPlan, RetrievalResult
from librarian.schemas.state import LibrarianState
from core.logging import get_logger

log = get_logger(__name__)


def _grade_chunks(
    results: list[RetrievalResult],
    threshold: float = 0.1,
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
    """Stateless node: retrieve → deduplicate → grade.

    Expands the query using plan.query_variants (multi-query).
    Falls back to state["query"] / state["standalone_query"] when no plan.
    """

    name = "retriever"
    description = (
        "Multi-query expansion, parallel embedding, hybrid search, dedup, and grading"
    )

    def __init__(
        self,
        retriever: Retriever,
        embedder: Embedder,
        top_k: int = 10,
        relevance_threshold: float = 0.1,
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

        all_results: list[RetrievalResult] = list(cached_results)
        if missed_variants:
            # Embed all variants in parallel; tolerate partial failures
            embed_outcomes = await asyncio.gather(
                *(self._embedder.aembed_query(v) for v in missed_variants),
                return_exceptions=True,
            )
            # Pair variants with embeddings, skipping failures
            valid_pairs: list[tuple[str, list[float]]] = []
            for variant, outcome in zip(missed_variants, embed_outcomes):
                if isinstance(outcome, BaseException):
                    log.warning(
                        "retrieval.embed.failed",
                        variant=variant[:80],
                        error=str(outcome),
                    )
                    continue
                valid_pairs.append((variant, outcome))

            if valid_pairs:
                search_outcomes = await asyncio.gather(
                    *(
                        self._retriever.search(
                            query_text=variant,
                            query_vector=qv,
                            k=self._top_k,
                        )
                        for variant, qv in valid_pairs
                    ),
                    return_exceptions=True,
                )
                for (variant, _), outcome in zip(valid_pairs, search_outcomes):
                    if isinstance(outcome, BaseException):
                        log.warning(
                            "retrieval.search.failed",
                            variant=variant[:80],
                            error=str(outcome),
                        )
                        continue
                    all_results.extend(outcome)
                    if self._cache is not None:
                        self._cache.put(
                            variant, self._cache_strategy, self._top_k, outcome
                        )

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

    def as_node(
        self,
    ) -> Callable[[LibrarianState], Coroutine[Any, Any, dict[str, Any]]]:
        """Return a LangGraph-compatible async node function."""

        async def retrieve(state: LibrarianState) -> dict[str, Any]:
            return await self.run(state)

        return retrieve
