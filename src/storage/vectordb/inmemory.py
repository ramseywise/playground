from __future__ import annotations

from librarian.retrieval.scoring import cosine_similarity, term_overlap
from librarian.schemas.chunks import Chunk, GradedChunk
from librarian.schemas.retrieval import RetrievalResult


class InMemoryRetriever:
    """In-process retriever for unit tests — no Docker, no OpenSearch.

    Mirrors OpenSearch hybrid search behaviour so test results match production:

    - bm25_weight=0.0, vector_weight=1.0  → pure knn (raptor / bedrock mode)
    - bm25_weight=1.0, vector_weight=0.0  → pure keyword
    - mixed weights                        → weighted linear blend (normalised scores)

    The weights stored here are the same fields used by the real OpenSearch
    retriever, so the same LibrarySettings object drives both environments.
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

        use_vector = self.vector_weight > 0.0
        use_bm25 = self.bm25_weight > 0.0

        # Pure vector (raptor / bedrock mode) — skip BM25 entirely
        if use_vector and not use_bm25:
            ranked = sorted(
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
            return [
                RetrievalResult(chunk=item.chunk, score=item.score, source="vector")
                for item in ranked[:k]
            ]

        # Pure keyword — skip vector entirely
        if use_bm25 and not use_vector:
            ranked = sorted(
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
            return [
                RetrievalResult(chunk=item.chunk, score=item.score, source="keyword")
                for item in ranked[:k]
            ]

        # Hybrid — weighted linear blend (mirrors OpenSearch normalised hybrid score)
        vector_scores = {
            chunk.id: max(
                0.0,
                cosine_similarity(query_vector, chunk.embedding)
                if chunk.embedding and query_vector
                else 0.0,
            )
            for chunk in candidates
        }
        kw_scores = {chunk.id: term_overlap(query_text, chunk.text) for chunk in candidates}

        def _norm(scores: dict[str, float]) -> dict[str, float]:
            mx = max(scores.values(), default=1.0) or 1.0
            return {cid: s / mx for cid, s in scores.items()}

        vec_norm = _norm(vector_scores)
        kw_norm = _norm(kw_scores)

        blended = sorted(
            (
                GradedChunk(
                    chunk=chunk,
                    score=self.vector_weight * vec_norm[chunk.id]
                    + self.bm25_weight * kw_norm[chunk.id],
                    relevant=True,
                )
                for chunk in candidates
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return [
            RetrievalResult(chunk=item.chunk, score=item.score, source="hybrid")
            for item in blended[:k]
        ]
