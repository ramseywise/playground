from __future__ import annotations

from agents.librarian.pipeline.retrieval.rrf import fuse_rankings
from agents.librarian.pipeline.retrieval.scoring import cosine_similarity, term_overlap
from agents.librarian.pipeline.schemas.chunks import Chunk, GradedChunk
from agents.librarian.pipeline.schemas.retrieval import RetrievalResult


class InMemoryRetriever:
    """In-process retriever for unit tests — no Docker, no OpenSearch.

    Hybrid score uses Reciprocal Rank Fusion over keyword and vector rankings.
    Mirrors the OpenSearch hybrid search interface so tests transfer directly.
    """

    def __init__(
        self,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> None:
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self._chunks: list[Chunk] = []

    async def upsert(self, chunks: list[Chunk]) -> None:
        existing_ids = {c.id for c in self._chunks}
        for chunk in chunks:
            if chunk.id in existing_ids:
                self._chunks = [c if c.id != chunk.id else chunk for c in self._chunks]
            else:
                self._chunks.append(chunk)

    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        candidates = self._chunks
        if metadata_filter:
            candidates = [
                c
                for c in candidates
                if all(
                    getattr(c.metadata, field, None) == value
                    for field, value in metadata_filter.items()
                )
            ]

        vector_rank = sorted(
            (
                GradedChunk(
                    chunk=chunk,
                    score=max(
                        0.0,
                        cosine_similarity(query_vector, chunk.embedding)
                        if chunk.embedding and query_vector
                        else 0.0,
                    ),
                    relevant=True,
                )
                for chunk in candidates
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        keyword_rank = sorted(
            (
                GradedChunk(
                    chunk=chunk,
                    score=term_overlap(query_text, chunk.text),
                    relevant=True,
                )
                for chunk in candidates
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        fused = fuse_rankings([keyword_rank, vector_rank], top_k=k)
        return [
            RetrievalResult(chunk=item.chunk, score=item.score, source="hybrid")
            for item in fused
        ]
