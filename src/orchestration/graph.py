from __future__ import annotations

from collections.abc import Callable, Coroutine, Hashable
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from core.clients.llm import LLMClient
from orchestration.query_understanding import (
    QueryAnalyzer,
    QueryRouter,
)
from orchestration.history import HistoryCondenser
from orchestration.nodes.generation import GenerationSubgraph
from orchestration.nodes.reranker import RerankerSubgraph
from orchestration.nodes.retrieval import RetrievalSubgraph
from librarian.retrieval.cache import RetrievalCache
from librarian.reranker.base import Reranker
from librarian.retrieval.base import Embedder, Retriever
from librarian.schemas.chunks import GradedChunk, RankedChunk
from librarian.schemas.state import LibrarianState
from core.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Node name constants
# ---------------------------------------------------------------------------

_ANALYZE = "analyze"
_CONDENSE = "condense"
_RETRIEVE = "retrieve"
_SNIPPET_RETRIEVE = "snippet_retrieve"
_RERANK = "rerank"
_GENERATE = "generate"
_GATE = "gate"


# ---------------------------------------------------------------------------
# Node functions (thin wrappers around subgraph/analyzer objects)
# ---------------------------------------------------------------------------


_SyncNode = Callable[[LibrarianState], dict[str, Any]]
_AsyncNode = Callable[[LibrarianState], Coroutine[Any, Any, dict[str, Any]]]


def _make_analyze_node(analyzer: QueryAnalyzer, *, max_variants: int = 3) -> _SyncNode:
    def analyze(state: LibrarianState) -> dict[str, Any]:
        query = state.get("standalone_query") or state.get("query", "")
        analysis = analyzer.analyze(query)
        return {
            "intent": analysis.intent.value,
            "retrieval_mode": analysis.retrieval_mode,
            "query_variants": analysis.expanded_terms[:max_variants]
            if analysis.expanded_terms
            else [],
        }

    return analyze


def _make_condense_node(condenser: HistoryCondenser) -> _AsyncNode:
    async def condense(state: LibrarianState) -> dict[str, Any]:
        return await condenser.condense(state)

    return condense


def _make_snippet_retrieve_node(snippet_retriever: Retriever) -> _AsyncNode:
    async def snippet_retrieve(state: LibrarianState) -> dict[str, Any]:
        """Keyword-based retrieval from the snippet DB, bypassing embedding + reranker."""
        query = state.get("standalone_query") or state.get("query", "")
        results = await snippet_retriever.search(
            query_text=query,
            query_vector=[],
            k=5,
        )
        graded = [
            GradedChunk(chunk=r.chunk, score=r.score, relevant=True) for r in results
        ]
        reranked = [
            RankedChunk(chunk=r.chunk, relevance_score=r.score, rank=i + 1)
            for i, r in enumerate(results)
        ]
        log.info(
            "graph.snippet_retrieve.done",
            query=query[:80],
            result_count=len(results),
        )
        return {
            "retrieved_chunks": results,
            "graded_chunks": graded,
            "reranked_chunks": reranked,
            "confidence_score": max((r.score for r in results), default=0.0),
        }

    return snippet_retrieve


def _make_retrieve_node(subgraph: RetrievalSubgraph) -> _AsyncNode:
    async def retrieve(state: LibrarianState) -> dict[str, Any]:
        result = await subgraph.run(state)
        retry_count = int(state.get("retry_count") or 0)
        return {**result, "retry_count": retry_count}

    return retrieve


def _make_rerank_node(subgraph: RerankerSubgraph) -> _AsyncNode:
    async def rerank(state: LibrarianState) -> dict[str, Any]:
        return await subgraph.run(state)

    return rerank


def _make_generate_node(subgraph: GenerationSubgraph) -> _AsyncNode:
    async def generate(state: LibrarianState) -> dict[str, Any]:
        return await subgraph.run(state)

    return generate


def _make_gate_node(subgraph: GenerationSubgraph) -> _SyncNode:
    def gate(state: LibrarianState) -> dict[str, Any]:
        result = subgraph.confidence_gate(state)
        # Increment retry_count here so the state update is persisted by LangGraph
        if result.get("fallback_requested"):
            result["retry_count"] = int(state.get("retry_count") or 0) + 1
        return result

    return gate


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


def _route_after_analyze(
    state: LibrarianState,
    has_snippet_retriever: bool,
) -> Literal["retrieve", "snippet_retrieve", "generate"]:
    """3-way routing after query analysis.

    - Direct intents (conversational, out_of_scope) → generate (no retrieval)
    - retrieval_mode == "snippet" and a snippet retriever is wired → snippet_retrieve
    - everything else → retrieve (dense/hybrid via vector store)
    """
    intent = state.get("intent", "lookup")
    if intent in ("conversational", "out_of_scope"):
        log.info("graph.route.direct", intent=intent)
        return "generate"
    if has_snippet_retriever and state.get("retrieval_mode") == "snippet":
        log.info("graph.route.snippet", intent=intent)
        return "snippet_retrieve"
    return "retrieve"


def _route_after_gate(
    state: LibrarianState,
    max_retries: int,
) -> Literal["generate", "retrieve"]:
    """CRAG loop: retry retrieval if not confident and under retry cap.

    retry_count at this point already reflects the increment applied by the gate node.
    """
    retry_count = int(state.get("retry_count") or 0)
    if state.get("fallback_requested") and retry_count <= max_retries:
        log.info("graph.crag.retry", retry=retry_count, max_retries=max_retries)
        return "retrieve"
    return "generate"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_graph(
    retriever: Retriever,
    embedder: Embedder,
    reranker: Reranker,
    llm: LLMClient,
    *,
    history_llm: LLMClient | None = None,
    history_condenser: HistoryCondenser | None = None,
    snippet_retriever: Retriever | None = None,
    cache: RetrievalCache | None = None,
    cache_strategy: str = "",
    retrieval_k: int = 10,
    reranker_top_k: int = 3,
    confidence_threshold: float = 0.3,
    max_crag_retries: int = 1,
    max_query_variants: int = 3,
) -> CompiledStateGraph:
    """Build and compile the LibrarianGraph.

    When *snippet_retriever* is provided, simple factual queries are routed to
    the snippet_retrieve node (keyword-based DuckDB FTS) instead of the vector
    store, bypassing embedding and reranking for fast lookup.

    Returns a compiled LangGraph runnable (``CompiledGraph``).
    """
    analyzer = QueryAnalyzer()
    router = QueryRouter()  # noqa: F841 — kept for future LLM routing integration
    condenser = history_condenser or HistoryCondenser(llm=history_llm or llm)

    retrieval_sg = RetrievalSubgraph(
        retriever=retriever,
        embedder=embedder,
        top_k=retrieval_k,
        cache=cache,
        cache_strategy=cache_strategy,
    )
    reranker_sg = RerankerSubgraph(reranker=reranker, top_k=reranker_top_k)
    generation_sg = GenerationSubgraph(
        llm=llm, confidence_threshold=confidence_threshold
    )

    has_snippet_retriever = snippet_retriever is not None
    graph = StateGraph(LibrarianState)

    # Register nodes
    graph.add_node(_CONDENSE, cast(Any, _make_condense_node(condenser)))
    graph.add_node(
        _ANALYZE,
        cast(Any, _make_analyze_node(analyzer, max_variants=max_query_variants)),
    )
    graph.add_node(_RETRIEVE, cast(Any, _make_retrieve_node(retrieval_sg)))
    graph.add_node(_RERANK, cast(Any, _make_rerank_node(reranker_sg)))
    graph.add_node(_GATE, cast(Any, _make_gate_node(generation_sg)))
    graph.add_node(_GENERATE, cast(Any, _make_generate_node(generation_sg)))

    if has_snippet_retriever:
        graph.add_node(
            _SNIPPET_RETRIEVE,
            cast(Any, _make_snippet_retrieve_node(snippet_retriever)),
        )
        graph.add_edge(_SNIPPET_RETRIEVE, _GENERATE)

    # Entry
    graph.add_edge(START, _CONDENSE)
    graph.add_edge(_CONDENSE, _ANALYZE)

    # After analyze: 3-way routing
    edge_map: dict[Hashable, str] = {
        "retrieve": _RETRIEVE,
        "generate": _GENERATE,
    }
    if has_snippet_retriever:
        edge_map["snippet_retrieve"] = _SNIPPET_RETRIEVE

    graph.add_conditional_edges(
        _ANALYZE,
        lambda s: _route_after_analyze(s, has_snippet_retriever),
        cast(dict[Hashable, str], edge_map),
    )

    # Dense/hybrid path: retrieve → rerank → gate
    graph.add_edge(_RETRIEVE, _RERANK)
    graph.add_edge(_RERANK, _GATE)

    # Gate: confident → generate; not confident + retries left → retrieve (CRAG)
    graph.add_conditional_edges(
        _GATE,
        lambda s: _route_after_gate(s, max_crag_retries),
        cast(dict[Hashable, str], {"generate": _GENERATE, "retrieve": _RETRIEVE}),
    )

    # Terminal
    graph.add_edge(_GENERATE, END)

    return graph.compile()
