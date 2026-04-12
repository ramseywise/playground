from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from librarian.reranker.base import Reranker
from librarian.schemas.chunks import GradedChunk, RankedChunk
from librarian.schemas.state import LibrarianState
from core.logging import get_logger

log = get_logger(__name__)

# confidence_score = max relevance_score among reranked chunks.
# If no chunks survive reranking the score is 0.0 (triggers CRAG retry).
_NO_CHUNKS_CONFIDENCE = 0.0


class RerankerAgent:
    """Stateless node: rerank graded_chunks → reranked_chunks + confidence_score.

    Only passes relevant graded chunks to the reranker.
    Falls back to all chunks when none are marked relevant (avoids empty rerank).
    """

    name = "reranker"
    description = "Cross-encoder or LLM-listwise reranking with confidence scoring"

    def __init__(self, reranker: Reranker, top_k: int = 3) -> None:
        self._reranker = reranker
        self._top_k = top_k

    async def run(self, state: LibrarianState) -> dict[str, Any]:
        graded: list[GradedChunk] = list(state.get("graded_chunks") or [])
        query = state.get("standalone_query") or state.get("query", "")

        if not graded:
            log.info("reranker.subgraph.skip", reason="no_graded_chunks")
            return {
                "reranked_chunks": [],
                "confidence_score": _NO_CHUNKS_CONFIDENCE,
            }

        # Prefer relevant chunks; fall back to all when none are flagged relevant
        candidates = [g for g in graded if g.relevant] or graded

        log.info(
            "reranker.subgraph.start",
            candidates=len(candidates),
            top_k=self._top_k,
            query=query,
        )

        try:
            reranked: list[RankedChunk] = await self._reranker.rerank(
                query, candidates, top_k=self._top_k
            )
        except Exception as exc:
            log.warning(
                "reranker.subgraph.error",
                error=str(exc),
                candidates=len(candidates),
                fallback="passthrough",
            )
            # Graceful degradation: return candidates in original order
            # with their retrieval scores as relevance, confidence 0 to trigger CRAG
            reranked = [
                RankedChunk(
                    chunk=g.chunk,
                    relevance_score=min(g.score, 1.0),
                    rank=i + 1,
                )
                for i, g in enumerate(candidates[: self._top_k])
            ]

        confidence = max(
            (r.relevance_score for r in reranked), default=_NO_CHUNKS_CONFIDENCE
        )

        log.info(
            "reranker.subgraph.done",
            reranked_count=len(reranked),
            confidence_score=confidence,
        )

        return {
            "reranked_chunks": reranked,
            "confidence_score": confidence,
        }

    def as_node(
        self,
    ) -> Callable[[LibrarianState], Coroutine[Any, Any, dict[str, Any]]]:
        """Return a LangGraph-compatible async node function."""

        async def rerank(state: LibrarianState) -> dict[str, Any]:
            return await self.run(state)

        return rerank


# Backward-compatible alias
RerankerSubgraph = RerankerAgent
