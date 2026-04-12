"""Capability tests: end-to-end pipeline behaviour over the golden corpus.

Tests here verify that the full graph (via create_librarian) produces
structurally correct outputs and routes correctly across intent types.
All LLM + reranker calls are mocked; retrieval uses InMemoryRetriever.
"""

from __future__ import annotations

import pytest

from librarian.tasks.models import GoldenSample
from librarian.factory import create_librarian
from storage.vectordb.inmemory import InMemoryRetriever
from tests.librarian.testing.mock_embedder import MockEmbedder
from librarian.config import LibrarySettings


# ---------------------------------------------------------------------------
# End-to-end: lookup queries produce response + citations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lookup_response_and_citations(
    eval_cfg: LibrarySettings,
    populated_retriever: InMemoryRetriever,
    eval_embedder: MockEmbedder,
    mock_llm_eval,
    mock_reranker_passthrough,
) -> None:
    graph = create_librarian(
        cfg=eval_cfg,
        llm=mock_llm_eval,
        embedder=eval_embedder,
        retriever=populated_retriever,
        reranker=mock_reranker_passthrough,
    )
    result = await graph.ainvoke(
        {"query": "how do I reset my API key?", "intent": "lookup"},
        config={"configurable": {"thread_id": "eval-1"}},
    )
    assert result["response"] == "eval answer"
    assert isinstance(result["citations"], list)


@pytest.mark.asyncio
async def test_citations_have_url_and_title(
    eval_cfg: LibrarySettings,
    populated_retriever: InMemoryRetriever,
    eval_embedder: MockEmbedder,
    mock_llm_eval,
    mock_reranker_passthrough,
) -> None:
    graph = create_librarian(
        cfg=eval_cfg,
        llm=mock_llm_eval,
        embedder=eval_embedder,
        retriever=populated_retriever,
        reranker=mock_reranker_passthrough,
    )
    result = await graph.ainvoke(
        {"query": "how do I reset my API key?", "intent": "lookup"},
        config={"configurable": {"thread_id": "eval-2"}},
    )
    for citation in result["citations"]:
        assert "url" in citation
        assert "title" in citation


@pytest.mark.asyncio
async def test_all_golden_queries_complete(
    golden_samples: list[GoldenSample],
    eval_cfg: LibrarySettings,
    populated_retriever: InMemoryRetriever,
    eval_embedder: MockEmbedder,
    mock_llm_eval,
    mock_reranker_passthrough,
) -> None:
    """All 5 golden queries produce a non-empty response without raising."""
    graph = create_librarian(
        cfg=eval_cfg,
        llm=mock_llm_eval,
        embedder=eval_embedder,
        retriever=populated_retriever,
        reranker=mock_reranker_passthrough,
    )
    for i, sample in enumerate(golden_samples):
        result = await graph.ainvoke(
            {"query": sample.query, "intent": "lookup"},
            config={"configurable": {"thread_id": f"eval-golden-{i}"}},
        )
        assert result["response"], f"Empty response for query_id={sample.query_id}"


# ---------------------------------------------------------------------------
# Capability: intent routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversational_handled_by_triage() -> None:
    """Conversational queries are intercepted by triage before the graph.

    This test verifies the triage module routes conversational intents to a
    direct response, so they never reach the librarian graph.
    """
    from interfaces.api.triage import TriageService

    decision = TriageService().decide("hello there")
    assert decision.route == "direct"
    assert decision.intent == "conversational"
    assert decision.response is not None


@pytest.mark.asyncio
async def test_out_of_scope_handled_by_triage() -> None:
    """Out-of-scope queries are intercepted by triage before the graph.

    This test verifies the triage module routes out_of_scope intents to an
    escalation response, so they never reach the librarian graph.
    """
    from interfaces.api.triage import TriageService

    decision = TriageService().decide("what is the weather?")
    assert decision.route == "escalation"
    assert decision.intent == "out_of_scope"
    assert decision.response is not None


# ---------------------------------------------------------------------------
# Capability: CRAG loop terminates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crag_loop_terminates_with_max_retries(
    eval_cfg: LibrarySettings,
    eval_embedder: MockEmbedder,
    mock_llm_eval,
) -> None:
    """Even with persistent low confidence, graph terminates after max_retries."""
    from unittest.mock import AsyncMock, MagicMock

    empty_retriever = MagicMock()
    empty_retriever.search = AsyncMock(return_value=[])

    empty_reranker = MagicMock()
    empty_reranker.rerank = AsyncMock(return_value=[])

    cfg = eval_cfg.model_copy(
        update={"confidence_threshold": 0.9, "max_crag_retries": 2}
    )
    graph = create_librarian(
        cfg=cfg,
        llm=mock_llm_eval,
        embedder=eval_embedder,
        retriever=empty_retriever,
        reranker=empty_reranker,
    )
    result = await graph.ainvoke(
        {"query": "auth question", "intent": "lookup"},
        config={"configurable": {"thread_id": "eval-crag"}},
    )
    # Graph must terminate: max 3 total search calls (initial + 2 retries)
    assert empty_retriever.search.call_count <= 3
    assert result["response"] == "eval answer"


# ---------------------------------------------------------------------------
# Capability: pipeline trace fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_output_has_required_state_keys(
    eval_cfg: LibrarySettings,
    populated_retriever: InMemoryRetriever,
    eval_embedder: MockEmbedder,
    mock_llm_eval,
    mock_reranker_passthrough,
) -> None:
    graph = create_librarian(
        cfg=eval_cfg,
        llm=mock_llm_eval,
        embedder=eval_embedder,
        retriever=populated_retriever,
        reranker=mock_reranker_passthrough,
    )
    result = await graph.ainvoke(
        {"query": "what is the rate limit?", "intent": "lookup"},
        config={"configurable": {"thread_id": "eval-state-keys"}},
    )
    for key in ("response", "citations", "intent", "graded_chunks", "reranked_chunks"):
        assert key in result, f"Missing state key: {key}"
