"""Tests for EnsembleRetriever (retrieval/ensemble.py)."""

from __future__ import annotations

import pytest

from librarian.retrieval.ensemble import EnsembleRetriever, _fingerprint
from librarian.schemas.chunks import Chunk, ChunkMetadata, GradedChunk
from librarian.schemas.queries import RetrievalResult
from storage.vectordb.inmemory import InMemoryRetriever
from tests.librarian.testing.mock_embedder import MockEmbedder


def _chunk(id_: str, text: str = "some chunk text", url: str = "https://example.com") -> Chunk:
    return Chunk(
        id=id_,
        text=text,
        metadata=ChunkMetadata(url=url, title="T", doc_id="d1"),
    )


def _seeded_chunks(n: int = 5) -> list[Chunk]:
    """Create *n* distinct chunks with embeddings already set."""
    embedder = MockEmbedder(dim=8, seed=99)
    chunks = []
    for i in range(n):
        text = f"chunk number {i} with unique content"
        c = _chunk(f"c{i}", text=text, url=f"https://example.com/doc{i}")
        c.embedding = embedder.embed_passage(text)
        chunks.append(c)
    return chunks


@pytest.fixture()
def embedder() -> MockEmbedder:
    return MockEmbedder(dim=8, seed=42)


@pytest.fixture()
def retriever() -> InMemoryRetriever:
    return InMemoryRetriever()


class TestEnsembleRetriever:
    @pytest.mark.asyncio()
    async def test_empty_queries_returns_empty(
        self, retriever: InMemoryRetriever, embedder: MockEmbedder
    ) -> None:
        ensemble = EnsembleRetriever([retriever], embedder)
        result = await ensemble.retrieve([])
        assert result == []

    @pytest.mark.asyncio()
    async def test_single_query_single_retriever(
        self, retriever: InMemoryRetriever, embedder: MockEmbedder
    ) -> None:
        chunks = _seeded_chunks(3)
        await retriever.upsert(chunks)

        ensemble = EnsembleRetriever(
            [retriever], embedder, score_threshold=0.0
        )
        result = await ensemble.retrieve(["chunk number 0"], k=10)
        assert len(result) > 0
        assert all(isinstance(gc, GradedChunk) for gc in result)

    @pytest.mark.asyncio()
    async def test_multi_query_returns_results(
        self, retriever: InMemoryRetriever, embedder: MockEmbedder
    ) -> None:
        chunks = _seeded_chunks(5)
        await retriever.upsert(chunks)

        ensemble = EnsembleRetriever(
            [retriever], embedder, score_threshold=0.0
        )
        result = await ensemble.retrieve(
            ["chunk number 0", "chunk number 3"], k=10
        )
        assert len(result) > 0

    @pytest.mark.asyncio()
    async def test_multi_retriever_fuses_results(
        self, embedder: MockEmbedder
    ) -> None:
        r1 = InMemoryRetriever(bm25_weight=0.0, vector_weight=1.0)
        r2 = InMemoryRetriever(bm25_weight=1.0, vector_weight=0.0)

        chunks = _seeded_chunks(5)
        await r1.upsert(chunks)
        await r2.upsert(chunks)

        ensemble = EnsembleRetriever(
            [r1, r2], embedder, score_threshold=0.0
        )
        result = await ensemble.retrieve(["chunk number 1"], k=10)
        assert len(result) > 0

    @pytest.mark.asyncio()
    async def test_score_threshold_filters(
        self, retriever: InMemoryRetriever, embedder: MockEmbedder
    ) -> None:
        chunks = _seeded_chunks(3)
        await retriever.upsert(chunks)

        # Very high threshold — should filter most/all results
        ensemble = EnsembleRetriever(
            [retriever], embedder, score_threshold=100.0
        )
        result = await ensemble.retrieve(["chunk number 0"], k=10)
        assert len(result) == 0

    @pytest.mark.asyncio()
    async def test_fingerprint_dedup_removes_duplicates(
        self, embedder: MockEmbedder
    ) -> None:
        """Two retrievers returning the same chunks → deduped to one copy each."""
        r1 = InMemoryRetriever()
        r2 = InMemoryRetriever()
        chunks = _seeded_chunks(2)
        await r1.upsert(chunks)
        await r2.upsert(chunks)

        ensemble = EnsembleRetriever(
            [r1, r2], embedder, score_threshold=0.0
        )
        result = await ensemble.retrieve(["chunk number 0"], k=10)
        # Should have at most len(chunks) results, not 2× (from 2 retrievers)
        assert len(result) <= len(chunks)

    @pytest.mark.asyncio()
    async def test_results_sorted_by_score_desc(
        self, retriever: InMemoryRetriever, embedder: MockEmbedder
    ) -> None:
        chunks = _seeded_chunks(5)
        await retriever.upsert(chunks)

        ensemble = EnsembleRetriever(
            [retriever], embedder, score_threshold=0.0
        )
        result = await ensemble.retrieve(["chunk number 2"], k=10)
        scores = [gc.score for gc in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio()
    async def test_relevant_flag_matches_threshold(
        self, retriever: InMemoryRetriever, embedder: MockEmbedder
    ) -> None:
        chunks = _seeded_chunks(3)
        await retriever.upsert(chunks)

        threshold = 0.01
        ensemble = EnsembleRetriever(
            [retriever], embedder, score_threshold=threshold
        )
        result = await ensemble.retrieve(["chunk number 0"], k=10)
        for gc in result:
            assert gc.relevant == (gc.score >= threshold)

    def test_requires_at_least_one_retriever(self, embedder: MockEmbedder) -> None:
        with pytest.raises(ValueError, match="at least one retriever"):
            EnsembleRetriever([], embedder)

    @pytest.mark.asyncio()
    async def test_custom_rrf_k(
        self, retriever: InMemoryRetriever, embedder: MockEmbedder
    ) -> None:
        chunks = _seeded_chunks(3)
        await retriever.upsert(chunks)

        e1 = EnsembleRetriever(
            [retriever], embedder, score_threshold=0.0, rrf_k=10
        )
        e2 = EnsembleRetriever(
            [retriever], embedder, score_threshold=0.0, rrf_k=60
        )
        r1 = await e1.retrieve(["chunk"], k=10)
        r2 = await e2.retrieve(["chunk"], k=10)
        # Smaller k → higher scores
        if r1 and r2:
            assert r1[0].score > r2[0].score


class TestFingerprint:
    def test_deterministic(self) -> None:
        r = RetrievalResult(chunk=_chunk("a"), score=0.5, source="hybrid")
        assert _fingerprint(r) == _fingerprint(r)

    def test_same_text_same_url_matches(self) -> None:
        r1 = RetrievalResult(chunk=_chunk("a", "hello"), score=0.5, source="hybrid")
        r2 = RetrievalResult(chunk=_chunk("b", "hello"), score=0.9, source="vector")
        assert _fingerprint(r1) == _fingerprint(r2)

    def test_different_url_differs(self) -> None:
        c1 = _chunk("a", "hello", url="https://a.com")
        c2 = _chunk("a", "hello", url="https://b.com")
        r1 = RetrievalResult(chunk=c1, score=0.5, source="hybrid")
        r2 = RetrievalResult(chunk=c2, score=0.5, source="hybrid")
        assert _fingerprint(r1) != _fingerprint(r2)

    def test_different_text_differs(self) -> None:
        r1 = RetrievalResult(chunk=_chunk("a", "hello"), score=0.5, source="hybrid")
        r2 = RetrievalResult(chunk=_chunk("a", "world"), score=0.5, source="hybrid")
        assert _fingerprint(r1) != _fingerprint(r2)
