"""Multi-query, multi-retriever fusion with fingerprint dedup.

``EnsembleRetriever`` parallelises embedding and search across query
variants, deduplicates results using a content fingerprint, applies
RRF fusion, and returns ``GradedChunk`` objects ready for reranking.
"""

from __future__ import annotations

import asyncio
from hashlib import sha256

from core.logging import get_logger
from librarian.retrieval.base import Embedder, Retriever
from librarian.retrieval.rrf import fuse_rankings
from librarian.schemas.chunks import GradedChunk
from librarian.schemas.queries import RetrievalResult

log = get_logger(__name__)


def _fingerprint(result: RetrievalResult) -> str:
    """Content-based fingerprint: ``url | text[:200]`` → SHA-256 prefix.

    Catches exact duplicates and near-duplicates from the same source
    (different chunk IDs pointing at overlapping text).
    """
    raw = f"{result.chunk.metadata.url}|{result.chunk.text[:200].lower().strip()}"
    return sha256(raw.encode()).hexdigest()[:16]


class EnsembleRetriever:
    """Multi-query, multi-retriever fusion with fingerprint dedup.

    Runs all *queries × retrievers* searches in parallel, deduplicates
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

    async def retrieve(
        self,
        queries: list[str],
        k: int = 10,
    ) -> list[GradedChunk]:
        """Run parallel retrieval across all queries and retrievers.

        Returns deduped, RRF-fused, threshold-filtered ``GradedChunk`` list
        sorted by score descending.
        """
        if not queries:
            return []

        # 1. Embed all queries in parallel
        embed_outcomes = await asyncio.gather(
            *(self._embedder.aembed_query(q) for q in queries),
            return_exceptions=True,
        )

        valid_pairs: list[tuple[str, list[float]]] = []
        for query, outcome in zip(queries, embed_outcomes):
            if isinstance(outcome, BaseException):
                log.warning(
                    "ensemble.embed.failed",
                    query=query[:80],
                    error=str(outcome),
                )
                continue
            valid_pairs.append((query, outcome))

        if not valid_pairs:
            log.warning("ensemble.all_embeddings_failed", query_count=len(queries))
            return []

        # 2. Search all (query, retriever) pairs in parallel
        search_tasks: list[asyncio.Task[list[RetrievalResult]]] = []
        for query_text, query_vector in valid_pairs:
            for retriever in self._retrievers:
                search_tasks.append(
                    asyncio.ensure_future(
                        retriever.search(
                            query_text=query_text,
                            query_vector=query_vector,
                            k=k,
                        )
                    )
                )

        search_outcomes = await asyncio.gather(*search_tasks, return_exceptions=True)

        # 3. Collect per-query ranked lists for RRF
        per_list_results: list[list[GradedChunk]] = []
        for i, outcome in enumerate(search_outcomes):
            if isinstance(outcome, BaseException):
                log.warning("ensemble.search.failed", task=i, error=str(outcome))
                continue
            graded = [
                GradedChunk(
                    chunk=r.chunk,
                    score=r.score,
                    relevant=r.score >= self._score_threshold,
                )
                for r in outcome
            ]
            if graded:
                per_list_results.append(graded)

        if not per_list_results:
            return []

        # 4. RRF fusion across all result lists
        fused = fuse_rankings(per_list_results, k=self._rrf_k)

        # 5. Fingerprint dedup — keep highest-scored copy
        seen: dict[str, int] = {}
        deduped: list[GradedChunk] = []
        for gc in fused:
            fp = _fingerprint(
                RetrievalResult(
                    chunk=gc.chunk,
                    score=gc.score,
                    source="hybrid",
                )
            )
            if fp in seen:
                existing = deduped[seen[fp]]
                if gc.score > existing.score:
                    deduped[seen[fp]] = gc
                continue
            seen[fp] = len(deduped)
            deduped.append(gc)

        # 6. Threshold filter
        filtered = [gc for gc in deduped if gc.score >= self._score_threshold]

        # Re-mark relevance after fusion (scores changed)
        for gc in filtered:
            gc.relevant = gc.score >= self._score_threshold

        log.info(
            "ensemble.done",
            queries=len(queries),
            retrievers=len(self._retrievers),
            raw_lists=len(per_list_results),
            fused=len(fused),
            deduped=len(deduped),
            filtered=len(filtered),
        )

        return filtered
