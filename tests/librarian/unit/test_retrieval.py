from __future__ import annotations

import pytest

from agents.librarian.retrieval.base import Embedder, Retriever
from agents.librarian.retrieval.inmemory import InMemoryRetriever
from agents.librarian.retrieval.mock_embedder import MockEmbedder
from agents.librarian.schemas.chunks import Chunk, ChunkMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(text: str, chunk_id: str = "c1", namespace: str | None = None) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            url="https://example.com",
            title="Doc",
            doc_id="doc-1",
            namespace=namespace,
        ),
    )


def _chunk_with_vec(text: str, vec: list[float], chunk_id: str = "c1") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(url="https://x.com", title="T", doc_id="d1"),
        embedding=vec,
    )


# ---------------------------------------------------------------------------
# MockEmbedder
# ---------------------------------------------------------------------------


def test_mock_embedder_returns_correct_dim(mock_embedder: MockEmbedder) -> None:
    vec = mock_embedder.embed_query("what is X?")
    assert len(vec) == 1024


def test_mock_embedder_query_is_deterministic(mock_embedder: MockEmbedder) -> None:
    v1 = mock_embedder.embed_query("hello")
    v2 = mock_embedder.embed_query("hello")
    assert v1 == v2


def test_mock_embedder_passage_is_deterministic(mock_embedder: MockEmbedder) -> None:
    v1 = mock_embedder.embed_passage("some doc text")
    v2 = mock_embedder.embed_passage("some doc text")
    assert v1 == v2


def test_mock_embedder_query_differs_from_passage(mock_embedder: MockEmbedder) -> None:
    # query: / passage: prefixes are stripped before cache key — same text same vec
    # This is intentional for test simplicity; real E5 would differ
    vq = mock_embedder.embed_query("topic")
    vp = mock_embedder.embed_passage("topic")
    assert vq == vp  # same underlying text → same cached vec


def test_mock_embedder_different_texts_differ(mock_embedder: MockEmbedder) -> None:
    va = mock_embedder.embed_query("alpha")
    vb = mock_embedder.embed_query("beta")
    assert va != vb


def test_mock_embedder_embed_passages_batch(mock_embedder: MockEmbedder) -> None:
    vecs = mock_embedder.embed_passages(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(len(v) == 1024 for v in vecs)


def test_mock_embedder_satisfies_protocol(mock_embedder: MockEmbedder) -> None:
    assert isinstance(mock_embedder, Embedder)


# ---------------------------------------------------------------------------
# InMemoryRetriever — upsert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inmemory_upsert_and_count(inmemory_retriever: InMemoryRetriever) -> None:
    await inmemory_retriever.upsert([_chunk("hello world"), _chunk("foo bar", "c2")])
    assert len(inmemory_retriever._chunks) == 2


@pytest.mark.asyncio
async def test_inmemory_upsert_deduplicates(
    inmemory_retriever: InMemoryRetriever,
) -> None:
    c = _chunk("original text")
    await inmemory_retriever.upsert([c])
    updated = _chunk_with_vec("updated text", [0.1] * 1024, chunk_id="c1")
    await inmemory_retriever.upsert([updated])
    assert len(inmemory_retriever._chunks) == 1
    assert inmemory_retriever._chunks[0].text == "updated text"


# ---------------------------------------------------------------------------
# InMemoryRetriever — keyword search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inmemory_keyword_match_ranks_first(
    inmemory_retriever: InMemoryRetriever,
) -> None:
    await inmemory_retriever.upsert(
        [
            _chunk("authentication tokens expire after 24 hours", "c1"),
            _chunk("billing invoice payment method credit card", "c2"),
        ]
    )
    results = await inmemory_retriever.search(
        query_text="authentication tokens",
        query_vector=[0.0] * 1024,
        k=2,
    )
    assert results[0].chunk.id == "c1"


@pytest.mark.asyncio
async def test_inmemory_returns_at_most_k(
    inmemory_retriever: InMemoryRetriever,
) -> None:
    chunks = [_chunk(f"doc {i}", f"c{i}") for i in range(10)]
    await inmemory_retriever.upsert(chunks)
    results = await inmemory_retriever.search("doc", [0.0] * 1024, k=3)
    assert len(results) <= 3


# ---------------------------------------------------------------------------
# InMemoryRetriever — vector search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inmemory_vector_match_ranks_first() -> None:
    retriever = InMemoryRetriever(bm25_weight=0.0, vector_weight=1.0)
    target_vec = [1.0] + [0.0] * 1023
    other_vec = [0.0] * 1024
    other_vec[1] = 1.0

    await retriever.upsert(
        [
            _chunk_with_vec("irrelevant text A", target_vec, "target"),
            _chunk_with_vec("irrelevant text B", other_vec, "other"),
        ]
    )
    results = await retriever.search("anything", target_vec, k=2)
    assert results[0].chunk.id == "target"


# ---------------------------------------------------------------------------
# InMemoryRetriever — metadata filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inmemory_metadata_filter(inmemory_retriever: InMemoryRetriever) -> None:
    await inmemory_retriever.upsert(
        [
            _chunk("public doc", "c1", namespace="public"),
            _chunk("internal doc", "c2", namespace="internal"),
        ]
    )
    results = await inmemory_retriever.search(
        "doc", [0.0] * 1024, k=10, metadata_filter={"namespace": "public"}
    )
    assert all(r.chunk.metadata.namespace == "public" for r in results)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# InMemoryRetriever — protocol compliance
# ---------------------------------------------------------------------------


def test_inmemory_satisfies_retriever_protocol(
    inmemory_retriever: InMemoryRetriever,
) -> None:
    assert isinstance(inmemory_retriever, Retriever)


# ---------------------------------------------------------------------------
# Round-trip: upsert with embedder then search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_roundtrip_upsert_and_retrieve(
    inmemory_retriever: InMemoryRetriever,
    mock_embedder: MockEmbedder,
) -> None:
    texts = [
        "setup and installation guide for the library",
        "billing and payment frequently asked questions",
        "authentication using API keys and OAuth tokens",
    ]
    chunks = []
    for i, text in enumerate(texts):
        vec = mock_embedder.embed_passage(text)
        chunks.append(_chunk_with_vec(text, vec, f"c{i}"))
    await inmemory_retriever.upsert(chunks)

    query_vec = mock_embedder.embed_query("setup installation")
    results = await inmemory_retriever.search("setup installation", query_vec, k=3)

    assert len(results) >= 1
    # setup doc should rank first (keyword overlap + vector similarity)
    assert results[0].chunk.id == "c0"
