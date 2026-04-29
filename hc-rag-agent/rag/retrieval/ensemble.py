"""Multi-query, multi-retriever fusion with fingerprint dedup.

EnsembleRetriever parallelises embedding and search across query variants,
deduplicates results using a content fingerprint, applies RRF fusion,
and returns GradedChunk objects ready for reranking.
"""

from __future__ import annotations

import asyncio
import logging

from rag.protocols import Embedder, Retriever
from rag.retrieval.rrf import fuse_rankings
from rag.schemas.chunks import GradedChunk

log = logging.getLogger(__name__)


class EnsembleRetriever:
    """Multi-query, multi-retriever fusion with fingerprint dedup.

    Runs all queries × retrievers searches in parallel, deduplicates
    by content fingerprint (keeping the highest-scored copy), fuses
    with RRF, and filters by score threshold.
    """

    def __init__(
        self,
        retrievers: list[Retriever],
        embedder: Embedder,
        *,
        score_threshold: float = 0.4,
        rrf_k: int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("EnsembleRetriever requires at least one retriever")
        self._retrievers = retrievers
        self._embedder = embedder
        self._score_threshold = score_threshold
        self._rrf_k = rrf_k

    async def retrieve(self, queries: list[str], k: int = 10) -> list[GradedChunk]:
        """Run parallel retrieval across all queries and retrievers.

        Returns deduped, RRF-fused, threshold-filtered GradedChunk list
        sorted by score descending.
        """
        if not queries:
            return []

        embed_outcomes = await asyncio.gather(
            *(self._embedder.aembed_query(q) for q in queries),
            return_exceptions=True,
        )

        valid_pairs: list[tuple[str, list[float]]] = []
        for query, outcome in zip(queries, embed_outcomes):
            if isinstance(outcome, BaseException):
                log.warning(
                    "ensemble.embed.failed query=%s error=%s", query[:80], outcome
                )
                continue
            valid_pairs.append((query, outcome))

        if not valid_pairs:
            log.warning("ensemble.all_embeddings_failed query_count=%d", len(queries))
            return []

        search_tasks = [
            asyncio.create_task(
                retriever.search(query_text=query_text, query_vector=query_vector, k=k)
            )
            for query_text, query_vector in valid_pairs
            for retriever in self._retrievers
        ]

        search_outcomes = await asyncio.gather(*search_tasks, return_exceptions=True)

        per_list_results: list[list[GradedChunk]] = []
        for i, search_outcome in enumerate(search_outcomes):
            if isinstance(search_outcome, BaseException):
                log.warning(
                    "ensemble.search.failed task=%d error=%s", i, search_outcome
                )
                continue
            graded = [
                GradedChunk(
                    chunk=r.chunk,
                    score=r.score,
                    relevant=r.score >= self._score_threshold,
                )
                for r in search_outcome
            ]
            if graded:
                per_list_results.append(graded)

        if not per_list_results:
            return []

        # fuse_rankings already deduplicates by content fingerprint
        fused = fuse_rankings(per_list_results, k=self._rrf_k)

        filtered = [gc for gc in fused if gc.score >= self._score_threshold]
        for gc in filtered:
            gc.relevant = True

        log.info(
            "ensemble.done queries=%d retrievers=%d fused=%d filtered=%d",
            len(queries),
            len(self._retrievers),
            len(fused),
            len(filtered),
        )
        return filtered
