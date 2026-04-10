from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from agents.librarian.orchestration.graph import (
    build_graph,
    _route_after_analyze,
    _route_after_gate,
)
from agents.librarian.schemas.chunks import (
    Chunk,
    ChunkMetadata,
    GradedChunk,
    RankedChunk,
)
from agents.librarian.schemas.state import LibrarianState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(cid: str, text: str = "auth text") -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        metadata=ChunkMetadata(url=f"https://x.com/{cid}", title="T", doc_id="d1"),
    )


def _ranked(cid: str, score: float = 0.9) -> RankedChunk:
    return RankedChunk(chunk=_chunk(cid), relevance_score=score, rank=1)


def _graded(cid: str, score: float = 0.8, relevant: bool = True) -> GradedChunk:
    return GradedChunk(chunk=_chunk(cid), score=score, relevant=relevant)


def _make_graph(
    retriever_results: list = [],
    reranker_results: list | None = None,
    llm_response: str = "the answer",
    confidence_threshold: float = 0.3,
    max_crag_retries: int = 1,
) -> object:
    if reranker_results is None:
        reranker_results = [_ranked("c1", 0.9)]

    mock_embedder = MagicMock()
    mock_embedder.embed_query = MagicMock(return_value=[0.1] * 64)

    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=retriever_results)

    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=reranker_results)

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value=llm_response)

    return build_graph(
        retriever=mock_retriever,
        embedder=mock_embedder,
        reranker=mock_reranker,
        llm=mock_llm,
        retrieval_k=5,
        reranker_top_k=3,
        confidence_threshold=confidence_threshold,
        max_crag_retries=max_crag_retries,
    )


# ---------------------------------------------------------------------------
# _route_after_analyze
# ---------------------------------------------------------------------------


def test_route_analyze_lookup_goes_retrieve() -> None:
    state: LibrarianState = {"query": "q", "intent": "lookup"}
    assert _route_after_analyze(state, has_snippet_retriever=False) == "retrieve"


def test_route_analyze_explore_goes_retrieve() -> None:
    state: LibrarianState = {"query": "q", "intent": "explore"}
    assert _route_after_analyze(state, has_snippet_retriever=False) == "retrieve"


def test_route_analyze_conversational_goes_generate() -> None:
    state: LibrarianState = {"query": "hi", "intent": "conversational"}
    assert _route_after_analyze(state, has_snippet_retriever=False) == "generate"


def test_route_analyze_out_of_scope_goes_generate() -> None:
    state: LibrarianState = {"query": "weather?", "intent": "out_of_scope"}
    assert _route_after_analyze(state, has_snippet_retriever=False) == "generate"


def test_route_analyze_missing_intent_defaults_retrieve() -> None:
    state: LibrarianState = {"query": "q"}
    assert _route_after_analyze(state, has_snippet_retriever=False) == "retrieve"


# ---------------------------------------------------------------------------
# _route_after_gate
# ---------------------------------------------------------------------------


def test_gate_confident_goes_generate() -> None:
    state: LibrarianState = {
        "query": "q",
        "confident": True,
        "fallback_requested": False,
    }
    assert _route_after_gate(state, max_retries=1) == "generate"


def test_gate_not_confident_under_limit_goes_retrieve() -> None:
    # retry_count=1 means gate node already incremented from 0; max_retries=1 → still retry
    state: LibrarianState = {"query": "q", "fallback_requested": True, "retry_count": 1}
    assert _route_after_gate(state, max_retries=1) == "retrieve"


def test_gate_not_confident_at_limit_goes_generate() -> None:
    # retry_count=2 after second gate increment; max_retries=1 → stop
    state: LibrarianState = {"query": "q", "fallback_requested": True, "retry_count": 2}
    assert _route_after_gate(state, max_retries=1) == "generate"


def test_gate_zero_retries_always_generate() -> None:
    # retry_count=1 after first gate increment; max_retries=0 → stop immediately
    state: LibrarianState = {"query": "q", "fallback_requested": True, "retry_count": 1}
    assert _route_after_gate(state, max_retries=0) == "generate"


# ---------------------------------------------------------------------------
# build_graph — integration (compiled graph invocation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_lookup_returns_response() -> None:
    graph = _make_graph()
    result = await graph.ainvoke({"query": "what is auth?", "intent": "lookup"})
    assert result["response"] == "the answer"


@pytest.mark.asyncio
async def test_graph_lookup_sets_intent() -> None:
    graph = _make_graph()
    result = await graph.ainvoke({"query": "what is auth?"})
    assert "intent" in result
    assert result["intent"] in (
        "lookup",
        "explore",
        "compare",
        "conversational",
        "out_of_scope",
    )


@pytest.mark.asyncio
async def test_graph_conversational_skips_retrieval() -> None:
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])
    mock_embedder = MagicMock()
    mock_embedder.embed_query = MagicMock(return_value=[0.1] * 64)
    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=[])
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="hello!")

    graph = build_graph(
        retriever=mock_retriever,
        embedder=mock_embedder,
        reranker=mock_reranker,
        llm=mock_llm,
    )
    result = await graph.ainvoke({"query": "hello", "intent": "conversational"})
    mock_retriever.search.assert_not_called()
    assert result["response"] == "hello!"


@pytest.mark.asyncio
async def test_graph_out_of_scope_skips_retrieval() -> None:
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])
    mock_embedder = MagicMock()
    mock_embedder.embed_query = MagicMock(return_value=[0.1] * 64)
    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=[])
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="out of scope")

    graph = build_graph(
        retriever=mock_retriever,
        embedder=mock_embedder,
        reranker=mock_reranker,
        llm=mock_llm,
    )
    result = await graph.ainvoke({"query": "stock price?", "intent": "out_of_scope"})
    mock_retriever.search.assert_not_called()
    assert result["response"] == "out of scope"


@pytest.mark.asyncio
async def test_graph_returns_citations() -> None:
    from agents.librarian.schemas.retrieval import RetrievalResult

    retrieval_result = RetrievalResult(
        chunk=_chunk("c1", "auth text"),
        score=0.8,
        source="hybrid",
    )
    graph = _make_graph(
        retriever_results=[retrieval_result],
        reranker_results=[_ranked("c1", 0.9)],
    )
    result = await graph.ainvoke({"query": "what is auth?", "intent": "lookup"})
    assert isinstance(result["citations"], list)


@pytest.mark.asyncio
async def test_graph_crag_retry_on_low_confidence() -> None:
    """Low confidence → one retry → then generate regardless."""
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])
    mock_embedder = MagicMock()
    mock_embedder.embed_query = MagicMock(return_value=[0.1] * 64)
    mock_reranker = MagicMock()
    # Returns empty → confidence = 0.0, below threshold=0.3
    mock_reranker.rerank = AsyncMock(return_value=[])
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="fallback answer")

    graph = build_graph(
        retriever=mock_retriever,
        embedder=mock_embedder,
        reranker=mock_reranker,
        llm=mock_llm,
        confidence_threshold=0.3,
        max_crag_retries=1,
    )
    result = await graph.ainvoke({"query": "what is auth?", "intent": "lookup"})
    # search called twice: initial + one CRAG retry
    assert mock_retriever.search.call_count == 2
    assert result["response"] == "fallback answer"


@pytest.mark.asyncio
async def test_graph_no_retry_when_max_retries_zero() -> None:
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])
    mock_embedder = MagicMock()
    mock_embedder.embed_query = MagicMock(return_value=[0.1] * 64)
    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=[])
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="no retry")

    graph = build_graph(
        retriever=mock_retriever,
        embedder=mock_embedder,
        reranker=mock_reranker,
        llm=mock_llm,
        confidence_threshold=0.3,
        max_crag_retries=0,
    )
    await graph.ainvoke({"query": "what is auth?", "intent": "lookup"})
    assert mock_retriever.search.call_count == 1


@pytest.mark.asyncio
async def test_graph_preserves_message_history() -> None:
    graph = _make_graph()
    history = [HumanMessage(content="previous question")]
    result = await graph.ainvoke(
        {
            "query": "follow-up",
            "intent": "lookup",
            "messages": history,
        }
    )
    assert result["response"] == "the answer"
