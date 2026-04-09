from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agents.librarian.orchestration.query_understanding import (
    QueryAnalyzer,
    QueryRouter,
)
from agents.librarian.orchestration.subgraphs.generation import GenerationSubgraph
from agents.librarian.orchestration.subgraphs.reranker import RerankerSubgraph
from agents.librarian.orchestration.subgraphs.retrieval import RetrievalSubgraph
from agents.librarian.reranker.base import Reranker
from agents.librarian.retrieval.base import Embedder, Retriever
from agents.librarian.schemas.state import LibrarianState
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Node name constants
# ---------------------------------------------------------------------------

_ANALYZE = "analyze"
_RETRIEVE = "retrieve"
_RERANK = "rerank"
_GENERATE = "generate"
_GATE = "gate"


# ---------------------------------------------------------------------------
# Node functions (thin wrappers around subgraph/analyzer objects)
# ---------------------------------------------------------------------------


def _make_analyze_node(analyzer: QueryAnalyzer) -> Any:
    def analyze(state: LibrarianState) -> dict[str, Any]:
        query = state.get("standalone_query") or state.get("query", "")
        analysis = analyzer.analyze(query)
        return {
            "intent": analysis.intent.value,
            "query_variants": analysis.expanded_terms[:3]
            if analysis.expanded_terms
            else [],
        }

    return analyze


def _make_retrieve_node(subgraph: RetrievalSubgraph) -> Any:
    async def retrieve(state: LibrarianState) -> dict[str, Any]:
        result = await subgraph.run(state)
        retry_count = int(state.get("retry_count") or 0)
        return {**result, "retry_count": retry_count}

    return retrieve


def _make_rerank_node(subgraph: RerankerSubgraph) -> Any:
    async def rerank(state: LibrarianState) -> dict[str, Any]:
        return await subgraph.run(state)

    return rerank


def _make_generate_node(subgraph: GenerationSubgraph) -> Any:
    async def generate(state: LibrarianState) -> dict[str, Any]:
        return await subgraph.run(state)

    return generate


def _make_gate_node(subgraph: GenerationSubgraph) -> Any:
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


def _route_after_analyze(state: LibrarianState) -> Literal["retrieve", "generate"]:
    """Direct intents skip retrieval entirely."""
    intent = state.get("intent", "lookup")
    if intent in ("conversational", "out_of_scope"):
        log.info("graph.route.direct", intent=intent)
        return "generate"
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
    llm: Any,
    *,
    retrieval_k: int = 10,
    reranker_top_k: int = 3,
    confidence_threshold: float = 0.3,
    max_crag_retries: int = 1,
) -> Any:
    """Build and compile the LibrarianGraph.

    Returns a compiled LangGraph runnable (``CompiledGraph``).
    """
    analyzer = QueryAnalyzer()
    router = QueryRouter()  # noqa: F841 — kept for future LLM routing integration

    retrieval_sg = RetrievalSubgraph(
        retriever=retriever, embedder=embedder, top_k=retrieval_k
    )
    reranker_sg = RerankerSubgraph(reranker=reranker, top_k=reranker_top_k)
    generation_sg = GenerationSubgraph(
        llm=llm, confidence_threshold=confidence_threshold
    )

    graph = StateGraph(LibrarianState)

    # Register nodes
    graph.add_node(_ANALYZE, _make_analyze_node(analyzer))
    graph.add_node(_RETRIEVE, _make_retrieve_node(retrieval_sg))
    graph.add_node(_RERANK, _make_rerank_node(reranker_sg))
    graph.add_node(_GATE, _make_gate_node(generation_sg))
    graph.add_node(_GENERATE, _make_generate_node(generation_sg))

    # Entry
    graph.add_edge(START, _ANALYZE)

    # After analyze: direct intents → generate, retrieval intents → retrieve
    graph.add_conditional_edges(
        _ANALYZE,
        _route_after_analyze,
        {"retrieve": _RETRIEVE, "generate": _GENERATE},
    )

    # Retrieval → rerank → gate
    graph.add_edge(_RETRIEVE, _RERANK)
    graph.add_edge(_RERANK, _GATE)

    # Gate: confident → generate; not confident + retries left → retrieve (CRAG)
    graph.add_conditional_edges(
        _GATE,
        lambda s: _route_after_gate(s, max_crag_retries),
        {"generate": _GENERATE, "retrieve": _RETRIEVE},
    )

    # Terminal
    graph.add_edge(_GENERATE, END)

    return graph.compile()
