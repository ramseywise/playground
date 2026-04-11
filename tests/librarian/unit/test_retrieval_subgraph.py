from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.librarian.orchestration.nodes.retrieval import (
    RetrievalSubgraph,
    _grade_chunks,
)
from agents.librarian.tools.storage.vectordb.inmemory import InMemoryRetriever
from tests.librarian.testing.mock_embedder import MockEmbedder
from agents.librarian.pipeline.schemas.chunks import Chunk, ChunkMetadata
from agents.librarian.pipeline.schemas.retrieval import Intent, QueryPlan, RetrievalResult
from agents.librarian.pipeline.schemas.state import LibrarianState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, text: str = "some text") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(url=f"https://x.com/{chunk_id}", title="T", doc_id="d1"),
    )


def _result(chunk_id: str, score: float, text: str = "some text") -> RetrievalResult:
    return RetrievalResult(chunk=_chunk(chunk_id, text), score=score, source="hybrid")


def _state(**kwargs: object) -> LibrarianState:
    base: LibrarianState = {"query": "what is auth?", "intent": "lookup"}
    base.update(kwargs)  # type: ignore[typeddict-item]
    return base


def _plan(variants: list[str]) -> QueryPlan:
    return QueryPlan(
        intent=Intent.LOOKUP,
        routing="retrieve",
        query_variants=variants,
        needs_clarification=False,
    )


# ---------------------------------------------------------------------------
# _grade_chunks
# ---------------------------------------------------------------------------


def test_grade_chunks_relevant_above_threshold() -> None:
    results = [_result("c1", score=0.8), _result("c2", score=0.05)]
    graded = _grade_chunks(results, threshold=0.1)
    assert graded[0].relevant is True
    assert graded[1].relevant is False


def test_grade_chunks_deduplicates_by_id() -> None:
    results = [_result("c1", 0.9), _result("c1", 0.5)]
    graded = _grade_chunks(results, threshold=0.1)
    assert len(graded) == 1
    assert graded[0].chunk.id == "c1"


def test_grade_chunks_preserves_score() -> None:
    results = [_result("c1", score=0.42)]
    graded = _grade_chunks(results, threshold=0.1)
    assert graded[0].score == pytest.approx(0.42)


def test_grade_chunks_empty() -> None:
    assert _grade_chunks([], threshold=0.1) == []


def test_grade_chunks_exact_threshold_is_relevant() -> None:
    results = [_result("c1", score=0.1)]
    graded = _grade_chunks(results, threshold=0.1)
    assert graded[0].relevant is True


# ---------------------------------------------------------------------------
# RetrievalSubgraph.run — happy paths
# ---------------------------------------------------------------------------


@pytest.fixture()
def embedder() -> MockEmbedder:
    return MockEmbedder(dim=64, seed=0)


@pytest.fixture()
def subgraph(
    embedder: MockEmbedder, inmemory_retriever: InMemoryRetriever
) -> RetrievalSubgraph:
    return RetrievalSubgraph(
        retriever=inmemory_retriever,
        embedder=embedder,
        top_k=5,
    )


@pytest.mark.asyncio
async def test_run_returns_required_keys(
    subgraph: RetrievalSubgraph,
    inmemory_retriever: InMemoryRetriever,
    embedder: MockEmbedder,
) -> None:
    await inmemory_retriever.upsert([_chunk("c1", "authentication token")])
    result = await subgraph.run(_state())
    assert "retrieved_chunks" in result
    assert "graded_chunks" in result
    assert "query_variants" in result


@pytest.mark.asyncio
async def test_run_single_query_no_plan(
    subgraph: RetrievalSubgraph,
    inmemory_retriever: InMemoryRetriever,
) -> None:
    await inmemory_retriever.upsert([_chunk("c1", "authentication")])
    result = await subgraph.run(_state(query="auth"))
    assert result["query_variants"] == ["auth"]


@pytest.mark.asyncio
async def test_run_uses_standalone_query_over_query(
    subgraph: RetrievalSubgraph,
    inmemory_retriever: InMemoryRetriever,
) -> None:
    await inmemory_retriever.upsert([_chunk("c1", "auth")])
    result = await subgraph.run(_state(query="original", standalone_query="rewritten"))
    assert "rewritten" in result["query_variants"]


@pytest.mark.asyncio
async def test_run_expands_with_plan_variants(
    subgraph: RetrievalSubgraph,
    inmemory_retriever: InMemoryRetriever,
) -> None:
    await inmemory_retriever.upsert([_chunk("c1", "auth"), _chunk("c2", "token")])
    plan = _plan(["auth tokens", "api key authentication"])
    result = await subgraph.run(_state(plan=plan))
    # base query prepended, plus variants = 3 total
    assert len(result["query_variants"]) == 3


@pytest.mark.asyncio
async def test_run_deduplicates_across_variants(
    subgraph: RetrievalSubgraph,
    inmemory_retriever: InMemoryRetriever,
) -> None:
    """Same chunk retrieved by multiple query variants must appear once in graded."""
    await inmemory_retriever.upsert([_chunk("c1", "authentication")])
    plan = _plan(["authentication", "auth", "login"])
    result = await subgraph.run(_state(plan=plan))
    ids = [g.chunk.id for g in result["graded_chunks"]]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_run_empty_index_returns_empty_graded(
    subgraph: RetrievalSubgraph,
) -> None:
    result = await subgraph.run(_state())
    assert result["graded_chunks"] == []
    assert result["retrieved_chunks"] == []


@pytest.mark.asyncio
async def test_run_graded_chunks_have_relevant_flag(
    subgraph: RetrievalSubgraph,
    inmemory_retriever: InMemoryRetriever,
) -> None:
    await inmemory_retriever.upsert([_chunk("c1", "what is auth")])
    result = await subgraph.run(_state(query="what is auth"))
    for gc in result["graded_chunks"]:
        assert isinstance(gc.relevant, bool)


@pytest.mark.asyncio
async def test_run_base_query_prepended_once_even_if_in_variants(
    subgraph: RetrievalSubgraph,
    inmemory_retriever: InMemoryRetriever,
) -> None:
    """base_query already in variants → not duplicated."""
    await inmemory_retriever.upsert([_chunk("c1")])
    plan = _plan(["what is auth?", "auth explanation"])
    result = await subgraph.run(_state(query="what is auth?", plan=plan))
    assert result["query_variants"].count("what is auth?") == 1


# ---------------------------------------------------------------------------
# RetrievalSubgraph.run — retriever called once per variant
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_calls_retriever_once_per_variant(embedder: MockEmbedder) -> None:
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])
    embedder.aembed_query = AsyncMock(wraps=embedder.aembed_query)  # type: ignore[method-assign]

    sg = RetrievalSubgraph(retriever=mock_retriever, embedder=embedder, top_k=5)
    plan = _plan(["v1", "v2"])
    await sg.run(_state(query="base", plan=plan))
    # base + v1 + v2 = 3 calls
    assert mock_retriever.search.call_count == 3
    assert embedder.aembed_query.call_count == 3


@pytest.mark.asyncio
async def test_run_calls_retriever_once_when_no_plan(embedder: MockEmbedder) -> None:
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])

    sg = RetrievalSubgraph(retriever=mock_retriever, embedder=embedder, top_k=5)
    await sg.run(_state(query="single query"))
    assert mock_retriever.search.call_count == 1


# ---------------------------------------------------------------------------
# RetrievalSubgraph — custom relevance threshold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_custom_threshold_all_irrelevant(
    inmemory_retriever: InMemoryRetriever,
    embedder: MockEmbedder,
) -> None:
    """With very high threshold, all chunks are irrelevant."""
    await inmemory_retriever.upsert([_chunk("c1", "auth")])
    sg = RetrievalSubgraph(
        retriever=inmemory_retriever,
        embedder=embedder,
        top_k=5,
        relevance_threshold=0.999,
    )
    result = await sg.run(_state(query="auth"))
    assert all(not g.relevant for g in result["graded_chunks"])
