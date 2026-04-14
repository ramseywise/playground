"""Tests for ADK tool functions wrapping the Librarian retrieval stack."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from librarian.schemas.chunks import Chunk, ChunkMetadata, GradedChunk, RankedChunk
from librarian.schemas.retrieval import RetrievalResult
from orchestration.google_adk.tools import (
    ToolDeps,
    _get_deps,
    analyze_query,
    condense_query,
    configure_tools,
    escalate,
    rerank_results,
    search_knowledge_base,
)
from orchestration.langgraph.history import CondenserAgent
from orchestration.langgraph.nodes.reranker import RerankerAgent
from orchestration.langgraph.nodes.retrieval import RetrieverAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, text: str, url: str = "https://example.com") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(url=url, title=f"Doc {chunk_id}", doc_id=chunk_id),
    )


def _make_retrieval_result(chunk_id: str, text: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk=_make_chunk(chunk_id, text),
        score=score,
        source="hybrid",
    )


def _make_ranked_chunk(
    chunk_id: str, text: str, score: float, rank: int
) -> RankedChunk:
    return RankedChunk(
        chunk=_make_chunk(chunk_id, text),
        relevance_score=score,
        rank=rank,
    )


def _make_retriever_agent(
    *,
    run_return: dict | None = None,
) -> RetrieverAgent:
    """Create a RetrieverAgent with mocked internals."""
    agent = RetrieverAgent(retriever=MagicMock(), embedder=MagicMock())
    agent.run = AsyncMock(return_value=run_return or {"graded_chunks": [], "retrieved_chunks": [], "query_variants": []})
    return agent


def _make_reranker_agent(
    *,
    run_return: dict | None = None,
) -> RerankerAgent:
    """Create a RerankerAgent with mocked internals."""
    agent = RerankerAgent(reranker=MagicMock())
    agent.run = AsyncMock(return_value=run_return or {"reranked_chunks": [], "confidence_score": 0.0})
    return agent


def _make_condenser_agent(
    *,
    condense_return: dict | None = None,
) -> CondenserAgent:
    """Create a CondenserAgent with mocked internals."""
    agent = CondenserAgent(llm=MagicMock())
    agent.condense = AsyncMock(return_value=condense_return or {"standalone_query": ""})
    return agent


def _configure_defaults(
    *,
    retriever_agent: RetrieverAgent | None = None,
    reranker_agent: RerankerAgent | None = None,
    condenser_agent: CondenserAgent | None = None,
) -> ToolDeps:
    """Configure tools with default mocked agents."""
    return configure_tools(
        retriever_agent=retriever_agent or _make_retriever_agent(),
        reranker_agent=reranker_agent or _make_reranker_agent(),
        condenser_agent=condenser_agent,
    )


# ---------------------------------------------------------------------------
# configure_tools
# ---------------------------------------------------------------------------


def test_get_deps_raises_when_not_configured() -> None:
    """Tools must raise if configure_tools() was never called."""
    import orchestration.google_adk.tools as tools_mod

    tools_mod._deps = None
    with pytest.raises(RuntimeError, match="not configured"):
        _get_deps()


def test_configure_tools_returns_deps() -> None:
    deps = _configure_defaults()
    assert isinstance(deps, ToolDeps)
    assert deps.retriever_agent is not None
    assert deps.analyzer is not None  # auto-created


def test_configure_tools_accepts_condenser_agent() -> None:
    condenser = _make_condenser_agent()
    deps = _configure_defaults(condenser_agent=condenser)
    assert deps.condenser_agent is condenser


# ---------------------------------------------------------------------------
# search_knowledge_base
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_knowledge_base_returns_results() -> None:
    graded = [
        GradedChunk(chunk=_make_chunk("c1", "First passage about auth."), score=0.95, relevant=True),
        GradedChunk(chunk=_make_chunk("c2", "Second passage about billing."), score=0.72, relevant=True),
    ]
    retriever_agent = _make_retriever_agent(
        run_return={"graded_chunks": graded, "retrieved_chunks": [], "query_variants": ["what is authentication?"]},
    )
    _configure_defaults(retriever_agent=retriever_agent)

    result = await search_knowledge_base("what is authentication?", num_results=5)

    assert result["total"] == 2
    assert len(result["results"]) == 2
    assert result["results"][0]["chunk_id"] == "c1"
    assert result["results"][0]["score"] == 0.95
    retriever_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_knowledge_base_respects_num_results() -> None:
    graded = [
        GradedChunk(chunk=_make_chunk(f"c{i}", f"Passage {i}"), score=0.9 - i * 0.1, relevant=True)
        for i in range(5)
    ]
    retriever_agent = _make_retriever_agent(
        run_return={"graded_chunks": graded, "retrieved_chunks": [], "query_variants": ["test"]},
    )
    _configure_defaults(retriever_agent=retriever_agent)

    result = await search_knowledge_base("test query", num_results=2)
    assert len(result["results"]) == 2
    assert result["total"] == 2


# ---------------------------------------------------------------------------
# rerank_results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_results_returns_ranked() -> None:
    reranked = [
        _make_ranked_chunk("c2", "Billing info", 0.92, 1),
        _make_ranked_chunk("c1", "Auth info", 0.71, 2),
    ]
    reranker_agent = _make_reranker_agent(
        run_return={"reranked_chunks": reranked, "confidence_score": 0.92},
    )
    _configure_defaults(reranker_agent=reranker_agent)

    passages = [
        {"text": "Auth info", "chunk_id": "c1", "url": "https://a.com", "title": "A", "score": "0.8"},
        {"text": "Billing info", "chunk_id": "c2", "url": "https://b.com", "title": "B", "score": "0.7"},
    ]

    result = await rerank_results("what is billing?", passages, top_k=2)

    assert len(result["results"]) == 2
    assert result["results"][0]["rank"] == 1
    assert result["confidence"] == 0.92
    reranker_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_rerank_results_empty_returns_zero_confidence() -> None:
    reranker_agent = _make_reranker_agent(
        run_return={"reranked_chunks": [], "confidence_score": 0.0},
    )
    _configure_defaults(reranker_agent=reranker_agent)

    result = await rerank_results("test", [], top_k=3)
    assert result["results"] == []
    assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# analyze_query
# ---------------------------------------------------------------------------


def test_analyze_query_returns_intent_and_entities() -> None:
    _configure_defaults()

    result = analyze_query("how does OAuth2 compare to SAML for authentication?")

    assert "intent" in result
    assert "entities" in result
    assert "complexity" in result
    assert "expanded_terms" in result
    assert "retrieval_mode" in result
    assert result["confidence"] > 0


def test_analyze_query_simple_query() -> None:
    _configure_defaults()

    result = analyze_query("what is OAuth?")
    assert result["complexity"] in ("simple", "moderate")


# ---------------------------------------------------------------------------
# condense_query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_condense_query_single_turn_passes_through() -> None:
    """Single-turn queries should not be rewritten."""
    _configure_defaults()

    result = await condense_query("what is OAuth?", [])
    assert result["standalone_query"] == "what is OAuth?"
    assert result["was_rewritten"] is False


@pytest.mark.asyncio
async def test_condense_query_no_agent_passes_through() -> None:
    """Without condenser_agent configured, should pass through."""
    _configure_defaults(condenser_agent=None)

    history = [
        {"role": "user", "content": "what is OAuth?"},
        {"role": "assistant", "content": "OAuth is a protocol..."},
        {"role": "user", "content": "how about SAML?"},
    ]
    result = await condense_query("how about SAML?", history)
    assert result["standalone_query"] == "how about SAML?"
    assert result["was_rewritten"] is False


@pytest.mark.asyncio
async def test_condense_query_rewrites_with_agent() -> None:
    """With condenser_agent, should rewrite the query."""
    condenser = _make_condenser_agent(
        condense_return={"standalone_query": "How does SAML compare to OAuth for authentication?"},
    )
    _configure_defaults(condenser_agent=condenser)

    history = [
        {"role": "user", "content": "what is OAuth?"},
        {"role": "assistant", "content": "OAuth is a protocol..."},
        {"role": "user", "content": "how about SAML?"},
    ]
    result = await condense_query("how about SAML?", history)
    assert (
        result["standalone_query"]
        == "How does SAML compare to OAuth for authentication?"
    )
    assert result["was_rewritten"] is True
    condenser.condense.assert_awaited_once()


# ---------------------------------------------------------------------------
# escalate
# ---------------------------------------------------------------------------


def test_escalate_out_of_scope() -> None:
    _configure_defaults()

    result = escalate(
        reason="out_of_scope",
        query="what's the weather today?",
    )

    assert result["escalated"] is True
    assert result["reason"] == "out_of_scope"
    assert "outside the knowledge base" in result["message"]
    assert result["reviewer_context"]["query"] == "what's the weather today?"


def test_escalate_low_confidence() -> None:
    _configure_defaults()

    result = escalate(
        reason="low_confidence",
        query="what is quantum entanglement?",
        context="Searched 3 times, max confidence 0.15",
    )

    assert result["escalated"] is True
    assert result["reason"] == "low_confidence"
    assert "human reviewer" in result["message"]
    assert (
        result["reviewer_context"]["context"] == "Searched 3 times, max confidence 0.15"
    )


def test_escalate_explicit_request() -> None:
    _configure_defaults()

    result = escalate(reason="explicit_request", query="let me talk to a human")
    assert result["escalated"] is True
    assert "connecting" in result["message"].lower()


def test_escalate_unknown_reason_falls_back() -> None:
    """Unknown reason should use out_of_scope message as fallback."""
    _configure_defaults()

    result = escalate(reason="something_else", query="test")
    assert result["escalated"] is True
    assert "outside the knowledge base" in result["message"]
