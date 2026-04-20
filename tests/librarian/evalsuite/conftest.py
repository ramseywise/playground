from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from librarian.ingestion.tasks.models import GoldenSample
from storage.vectordb.inmemory import InMemoryRetriever
from tests.librarian.testing.mock_embedder import MockEmbedder
from librarian.schemas.chunks import Chunk, ChunkMetadata
from librarian.config import LibrarySettings


# ---------------------------------------------------------------------------
# Golden dataset — 5 samples, deterministic for regression tracking
# ---------------------------------------------------------------------------

GOLDEN: list[GoldenSample] = [
    GoldenSample(
        query_id="q1",
        query="how do I reset my API key?",
        expected_doc_url="https://docs.example.com/api-keys",
        relevant_chunk_ids=["c1"],
        category="auth",
    ),
    GoldenSample(
        query_id="q2",
        query="what is the rate limit for the API?",
        expected_doc_url="https://docs.example.com/rate-limits",
        relevant_chunk_ids=["c2"],
        category="api",
    ),
    GoldenSample(
        query_id="q3",
        query="how do I install the SDK?",
        expected_doc_url="https://docs.example.com/install",
        relevant_chunk_ids=["c3"],
        category="setup",
    ),
    GoldenSample(
        query_id="q4",
        query="what authentication methods are supported?",
        expected_doc_url="https://docs.example.com/auth",
        relevant_chunk_ids=["c4"],
        category="auth",
    ),
    GoldenSample(
        query_id="q5",
        query="how do I handle errors in the API?",
        expected_doc_url="https://docs.example.com/errors",
        relevant_chunk_ids=["c5"],
        category="api",
    ),
]

# Corpus aligned to golden — each chunk's text contains keywords matching its query
CORPUS: list[Chunk] = [
    Chunk(
        id="c1",
        text="reset API key: navigate to settings and click regenerate",
        metadata=ChunkMetadata(
            url="https://docs.example.com/api-keys", title="API Keys", doc_id="d1"
        ),
    ),
    Chunk(
        id="c2",
        text="rate limit is 1000 requests per minute for the API",
        metadata=ChunkMetadata(
            url="https://docs.example.com/rate-limits", title="Rate Limits", doc_id="d2"
        ),
    ),
    Chunk(
        id="c3",
        text="install SDK using pip install or uv add package",
        metadata=ChunkMetadata(
            url="https://docs.example.com/install", title="Installation", doc_id="d3"
        ),
    ),
    Chunk(
        id="c4",
        text="authentication methods supported: API key, OAuth2, JWT",
        metadata=ChunkMetadata(
            url="https://docs.example.com/auth", title="Authentication", doc_id="d4"
        ),
    ),
    Chunk(
        id="c5",
        text="handle errors by catching exceptions and checking status codes",
        metadata=ChunkMetadata(
            url="https://docs.example.com/errors", title="Error Handling", doc_id="d5"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def golden_samples() -> list[GoldenSample]:
    return GOLDEN


@pytest.fixture()
def eval_embedder() -> MockEmbedder:
    return MockEmbedder(dim=64, seed=42)


@pytest_asyncio.fixture()
async def populated_retriever(eval_embedder: MockEmbedder) -> InMemoryRetriever:
    retriever = InMemoryRetriever(bm25_weight=0.6, vector_weight=0.4)
    chunks_with_embeddings = []
    for chunk in CORPUS:
        vec = eval_embedder.embed_passage(chunk.text)
        chunks_with_embeddings.append(chunk.model_copy(update={"embedding": vec}))
    await retriever.upsert(chunks_with_embeddings)
    return retriever


@pytest.fixture()
def eval_cfg() -> LibrarySettings:
    return LibrarySettings(
        retrieval_strategy="inmemory",
        reranker_strategy="cross_encoder",
        confidence_threshold=0.0,  # never block in eval
        max_crag_retries=0,
        retrieval_k=5,
        reranker_top_k=3,
        anthropic_api_key="test",
    )


@pytest.fixture()
def mock_llm_eval() -> MagicMock:
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="eval answer")
    return llm


@pytest.fixture()
def mock_reranker_passthrough() -> MagicMock:
    """Reranker that returns all chunks at score 0.8 — preserves retrieval rank."""
    from librarian.schemas.chunks import RankedChunk

    async def passthrough(query: str, chunks, top_k: int = 3):
        return [
            RankedChunk(chunk=g.chunk, relevance_score=0.8, rank=i + 1)
            for i, g in enumerate(chunks[:top_k])
        ]

    r = MagicMock()
    r.rerank = passthrough
    return r
