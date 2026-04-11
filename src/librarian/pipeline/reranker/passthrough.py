"""No-op passthrough reranker — skips reranking entirely.

Copies ``GradedChunk`` to ``RankedChunk`` without any re-scoring.
Useful for ablation studies and simple queries where reranking adds
latency without quality improvement.

Enable via ``reranker_strategy=passthrough`` in config.
"""

from __future__ import annotations

from agents.librarian.pipeline.schemas.chunks import GradedChunk, RankedChunk


class PassthroughReranker:
    """Passthrough — no reranking, just format conversion."""

    async def rerank(
        self,
        query: str,
        chunks: list[GradedChunk],
        top_k: int = 3,
    ) -> list[RankedChunk]:
        """Convert GradedChunks to RankedChunks without re-scoring.

        Uses the original retrieval score as the relevance_score.
        """
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]
        return [
            RankedChunk(
                chunk=gc.chunk,
                relevance_score=min(gc.score, 1.0),
                rank=i + 1,
            )
            for i, gc in enumerate(sorted_chunks)
        ]
