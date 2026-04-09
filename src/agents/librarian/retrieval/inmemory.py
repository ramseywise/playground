from __future__ import annotations

import math

from agents.librarian.schemas.chunks import Chunk
from agents.librarian.schemas.retrieval import RetrievalResult


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _term_overlap(query: str, text: str) -> float:
    """BM25-like term overlap: |query_terms ∩ doc_terms| / |query_terms|."""
    q_terms = set(query.lower().split())
    d_terms = set(text.lower().split())
    if not q_terms:
        return 0.0
    return len(q_terms & d_terms) / len(q_terms)


class InMemoryRetriever:
    """In-process retriever for unit tests — no Docker, no OpenSearch.

    Hybrid score = bm25_weight * term_overlap + vector_weight * cosine_similarity.
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

        scored: list[tuple[float, Chunk]] = []
        for chunk in candidates:
            vec_score = 0.0
            if chunk.embedding and query_vector:
                vec_score = max(0.0, _cosine(query_vector, chunk.embedding))
            kw_score = _term_overlap(query_text, chunk.text)
            score = self.bm25_weight * kw_score + self.vector_weight * vec_score
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalResult(chunk=chunk, score=score, source="hybrid")
            for score, chunk in scored[:k]
        ]
