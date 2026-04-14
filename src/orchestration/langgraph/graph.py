"""Librarian RAG Graph — CRAG pipeline with conditional routing.

Topology::

    START → condense → analyze →┬→ retrieve → rerank → gate →┬→ generate → END
                                │                             │
                                └→ snippet_retrieve ──────────┘→ retrieve (CRAG)

Agents:
    CondenserAgent   — rewrites multi-turn queries to standalone form (Haiku)
    PlannerAgent     — intent classification + query expansion (no LLM)
    RetrieverAgent   — multi-query embedding + hybrid search + grading
    RerankerAgent    — cross-encoder or LLM-listwise reranking
    GeneratorAgent   — prompt assembly + LLM generation + citation extraction
    QualityGate      — confidence threshold check for CRAG retry decision

Equivalent ADK structure (for reference)::

    SequentialAgent("librarian", sub_agents=[
        condenser, planner,
        LoopAgent("crag", sub_agents=[retriever, reranker, gate]),
        generator,
    ])
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Hashable
from typing import Any, Literal, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from clients.llm import LLMClient
from librarian.plan.analyzer import QueryAnalyzer
from orchestration.langgraph.history import CondenserAgent
from orchestration.langgraph.nodes.generation import GeneratorAgent
from orchestration.langgraph.nodes.reranker import RerankerAgent
from orchestration.langgraph.nodes.retrieval import RetrieverAgent
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
# Node functions (thin wrappers around agent objects)
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


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


def _route_after_analyze(
    state: LibrarianState,
    has_snippet_retriever: bool,
) -> Literal["retrieve", "snippet_retrieve"]:
    """2-way routing after query analysis.

    Triage handles conversational and out_of_scope before the graph is
    invoked.  This function decides between retrieval modes only.
    """
    if has_snippet_retriever and state.get("retrieval_mode") == "snippet":
        log.info("graph.route.snippet", intent=state.get("intent", ""))
        return "snippet_retrieve"
    return "retrieve"


def _route_after_gate(
    state: LibrarianState,
    max_retries: int,
) -> Literal["generate", "retrieve"]:
    """CRAG loop: retry retrieval if not confident and under retry cap.

    The gate node increments retry_count *before* this edge runs, so:
      max_crag_retries=1 → allows exactly 1 retry (retry_count goes 0→1, 1≤1=True)
      max_crag_retries=0 → no retries (retry_count goes 0→1, 1≤0=False)
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
    history_condenser: CondenserAgent | None = None,
    snippet_retriever: Retriever | None = None,
    cache: RetrievalCache | None = None,
    cache_strategy: str = "",
    retrieval_k: int = 10,
    reranker_top_k: int = 3,
    relevance_threshold: float = 0.1,
    confidence_threshold: float = 0.4,
    max_crag_retries: int = 1,
    max_query_variants: int = 3,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build and compile the LibrarianGraph.

    When *snippet_retriever* is provided, simple factual queries are routed to
    the snippet_retrieve node (keyword-based DuckDB FTS) instead of the vector
    store, bypassing embedding and reranking for fast lookup.

    Returns a compiled LangGraph runnable (``CompiledGraph``).
    """
    analyzer = QueryAnalyzer()
    condenser = history_condenser or CondenserAgent(llm=history_llm or llm)

    retrieval_agent = RetrieverAgent(
        retriever=retriever,
        embedder=embedder,
        top_k=retrieval_k,
        relevance_threshold=relevance_threshold,
        cache=cache,
        cache_strategy=cache_strategy,
    )
    reranker_agent = RerankerAgent(reranker=reranker, top_k=reranker_top_k)
    generator_agent = GeneratorAgent(llm=llm, confidence_threshold=confidence_threshold)

    has_snippet_retriever = snippet_retriever is not None
    graph = StateGraph(LibrarianState)

    # Register nodes — each agent wires itself via as_node()
    graph.add_node(_CONDENSE, cast(Any, condenser.as_node()))
    graph.add_node(
        _ANALYZE,
        cast(Any, _make_analyze_node(analyzer, max_variants=max_query_variants)),
    )
    graph.add_node(_RETRIEVE, cast(Any, retrieval_agent.as_node()))
    graph.add_node(_RERANK, cast(Any, reranker_agent.as_node()))
    graph.add_node(_GATE, cast(Any, generator_agent.as_gate_node()))
    graph.add_node(_GENERATE, cast(Any, generator_agent.as_node()))

    if has_snippet_retriever:
        graph.add_node(
            _SNIPPET_RETRIEVE,
            cast(Any, _make_snippet_retrieve_node(snippet_retriever)),
        )
        graph.add_edge(_SNIPPET_RETRIEVE, _GENERATE)

    # Entry
    graph.add_edge(START, _CONDENSE)
    graph.add_edge(_CONDENSE, _ANALYZE)

    # After analyze: 2-way routing (triage handles scope gating)
    edge_map: dict[Hashable, str] = {
        "retrieve": _RETRIEVE,
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

    return graph.compile(checkpointer=checkpointer)
