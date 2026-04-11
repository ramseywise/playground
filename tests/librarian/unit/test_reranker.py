from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.librarian.rag_core.reranker.base import Reranker
from agents.librarian.rag_core.reranker.cross_encoder import CrossEncoderReranker, _sigmoid
from agents.librarian.rag_core.reranker.llm_listwise import LLMListwiseReranker
from agents.librarian.rag_core.schemas.chunks import Chunk, ChunkMetadata, GradedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _graded(
    text: str, chunk_id: str, score: float = 0.5, relevant: bool = True
) -> GradedChunk:
    chunk = Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(url="https://x.com", title="T", doc_id="d1"),
    )
    return GradedChunk(chunk=chunk, score=score, relevant=relevant)


def _graded_list() -> list[GradedChunk]:
    return [
        _graded("authentication with API keys", "c1", score=0.9),
        _graded("billing and invoices FAQ", "c2", score=0.7),
        _graded("setup and installation guide", "c3", score=0.6),
        _graded("release notes version 2.0", "c4", score=0.3),
    ]


# ---------------------------------------------------------------------------
# _sigmoid
# ---------------------------------------------------------------------------


def test_sigmoid_zero() -> None:
    assert abs(_sigmoid(0.0) - 0.5) < 1e-6


def test_sigmoid_large_positive() -> None:
    assert _sigmoid(10.0) > 0.99


def test_sigmoid_large_negative() -> None:
    assert _sigmoid(-10.0) < 0.01


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------


@pytest.fixture()
def cross_encoder_reranker() -> CrossEncoderReranker:
    mock_model = MagicMock()
    import numpy as np

    # Return descending logits: c1 highest, c4 lowest
    mock_model.predict.return_value = np.array([3.0, 1.0, 0.5, -1.0])

    with patch(
        "agents.librarian.rag_core.reranker.cross_encoder._load_cross_encoder",
        return_value=mock_model,
    ):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model_name = "mock"
        reranker._model = mock_model
        return reranker


@pytest.mark.asyncio
async def test_cross_encoder_returns_top_k(
    cross_encoder_reranker: CrossEncoderReranker,
) -> None:
    results = await cross_encoder_reranker.rerank("auth query", _graded_list(), top_k=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_cross_encoder_sorted_desc(
    cross_encoder_reranker: CrossEncoderReranker,
) -> None:
    results = await cross_encoder_reranker.rerank("auth query", _graded_list(), top_k=3)
    scores = [r.relevance_score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_cross_encoder_scores_in_range(
    cross_encoder_reranker: CrossEncoderReranker,
) -> None:
    results = await cross_encoder_reranker.rerank("query", _graded_list(), top_k=4)
    assert all(0.0 <= r.relevance_score <= 1.0 for r in results)


@pytest.mark.asyncio
async def test_cross_encoder_rank_is_1_indexed(
    cross_encoder_reranker: CrossEncoderReranker,
) -> None:
    results = await cross_encoder_reranker.rerank("query", _graded_list(), top_k=3)
    assert [r.rank for r in results] == [1, 2, 3]


@pytest.mark.asyncio
async def test_cross_encoder_empty_input(
    cross_encoder_reranker: CrossEncoderReranker,
) -> None:
    results = await cross_encoder_reranker.rerank("query", [], top_k=3)
    assert results == []


def test_cross_encoder_satisfies_protocol(
    cross_encoder_reranker: CrossEncoderReranker,
) -> None:
    assert isinstance(cross_encoder_reranker, Reranker)


# ---------------------------------------------------------------------------
# LLMListwiseReranker — happy path
# ---------------------------------------------------------------------------


@pytest.fixture()
def llm_listwise(mock_llm: MagicMock) -> LLMListwiseReranker:
    return LLMListwiseReranker(llm=mock_llm)


@pytest.mark.asyncio
async def test_llm_listwise_happy_path(
    llm_listwise: LLMListwiseReranker, mock_llm: MagicMock
) -> None:
    mock_llm.generate = AsyncMock(
        return_value=(
            '[{"rank": 1, "doc_index": 0, "relevance_score": 0.95},'
            ' {"rank": 2, "doc_index": 2, "relevance_score": 0.80},'
            ' {"rank": 3, "doc_index": 1, "relevance_score": 0.60},'
            ' {"rank": 4, "doc_index": 3, "relevance_score": 0.30}]'
        )
    )
    results = await llm_listwise.rerank("auth", _graded_list(), top_k=3)
    assert len(results) == 3
    assert results[0].chunk.id == "c1"
    assert results[0].relevance_score == 0.95


@pytest.mark.asyncio
async def test_llm_listwise_top_k_limits(
    llm_listwise: LLMListwiseReranker, mock_llm: MagicMock
) -> None:
    mock_llm.generate = AsyncMock(
        return_value=(
            '[{"rank": 1, "doc_index": 0, "relevance_score": 0.9},'
            ' {"rank": 2, "doc_index": 1, "relevance_score": 0.8},'
            ' {"rank": 3, "doc_index": 2, "relevance_score": 0.7},'
            ' {"rank": 4, "doc_index": 3, "relevance_score": 0.5}]'
        )
    )
    results = await llm_listwise.rerank("query", _graded_list(), top_k=2)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# LLMListwiseReranker — fallback paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_listwise_total_parse_failure_fallback(
    llm_listwise: LLMListwiseReranker, mock_llm: MagicMock
) -> None:
    mock_llm.generate = AsyncMock(return_value="not valid json at all")
    results = await llm_listwise.rerank("query", _graded_list(), top_k=4)
    assert len(results) == 4
    assert all(r.relevance_score == 0.5 for r in results)


@pytest.mark.asyncio
async def test_llm_listwise_exception_fallback(
    llm_listwise: LLMListwiseReranker, mock_llm: MagicMock
) -> None:
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("API down"))
    results = await llm_listwise.rerank("query", _graded_list(), top_k=4)
    assert len(results) == 4
    assert all(r.relevance_score == 0.5 for r in results)


@pytest.mark.asyncio
async def test_llm_listwise_partial_parse_appends_missing(
    llm_listwise: LLMListwiseReranker, mock_llm: MagicMock
) -> None:
    # Only ranks 2 of 4 docs — missing c3, c4 should be appended at 0.5
    mock_llm.generate = AsyncMock(
        return_value=(
            '[{"rank": 1, "doc_index": 0, "relevance_score": 0.9},'
            ' {"rank": 2, "doc_index": 1, "relevance_score": 0.8}]'
        )
    )
    results = await llm_listwise.rerank("query", _graded_list(), top_k=10)
    assert len(results) == 4
    # First two from LLM, last two appended at 0.5
    assert results[0].relevance_score == 0.9
    assert results[2].relevance_score == 0.5
    assert results[3].relevance_score == 0.5


@pytest.mark.asyncio
async def test_llm_listwise_clamps_score_to_range(
    llm_listwise: LLMListwiseReranker, mock_llm: MagicMock
) -> None:
    mock_llm.generate = AsyncMock(
        return_value=(
            '[{"rank": 1, "doc_index": 0, "relevance_score": 2.5},'
            ' {"rank": 2, "doc_index": 1, "relevance_score": -0.3}]'
        )
    )
    results = await llm_listwise.rerank("query", _graded_list()[:2], top_k=2)
    assert all(0.0 <= r.relevance_score <= 1.0 for r in results)


@pytest.mark.asyncio
async def test_llm_listwise_strips_markdown_fences(
    llm_listwise: LLMListwiseReranker, mock_llm: MagicMock
) -> None:
    mock_llm.generate = AsyncMock(
        return_value='```json\n[{"rank": 1, "doc_index": 0, "relevance_score": 0.9}]\n```'
    )
    results = await llm_listwise.rerank("query", _graded_list()[:1], top_k=1)
    assert results[0].relevance_score == 0.9


def test_llm_listwise_satisfies_protocol(llm_listwise: LLMListwiseReranker) -> None:
    assert isinstance(llm_listwise, Reranker)
