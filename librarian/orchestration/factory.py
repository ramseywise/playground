"""LangGraph pipeline assembly + ingestion pipeline factory.

Composes components from ``orchestration.components`` into a compiled
LangGraph CRAG pipeline or an ``IngestionPipeline``.  Any component can
be overridden by passing it directly — the primary injection point for
tests and alternative configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph.state import CompiledStateGraph

from orchestration.components import (
    build_checkpointer,
    build_chunker,
    build_embedder,
    build_llm,
    build_history_llm,
    build_reranker,
    build_retriever,
    build_storage,
    warm_up_embedder,
)
from orchestration.langgraph.graph import build_graph
from orchestration.langgraph.history import CondenserAgent
from orchestration.langgraph.nodes.reranker import RerankerAgent
from orchestration.langgraph.nodes.retrieval import RetrieverAgent
from orchestration.langgraph.nodes.generation import GeneratorAgent
from librarian.retrieval.base import Embedder, Retriever
from librarian.reranker.base import Reranker
from librarian.retrieval.cache import RetrievalCache
from librarian.config import LibrarySettings, settings as _default_settings
from core.logging import get_logger

if TYPE_CHECKING:
    from clients.llm import LLMClient
    from librarian.ingestion.base import Chunker
    from librarian.ingestion.pipeline import IngestionPipeline

log = get_logger(__name__)

# Re-export warm_up_embedder so existing callers of orchestration.factory
# continue to work without changing their imports.
__all__ = [
    "create_agents",
    "create_ingestion_pipeline",
    "create_librarian",
    "warm_up_embedder",
]


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


def create_agents(
    cfg: LibrarySettings | None = None,
    *,
    llm: LLMClient | None = None,
    history_llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
) -> tuple[RetrieverAgent, RerankerAgent, GeneratorAgent, CondenserAgent]:
    """Build the canonical set of RAG agents from config.

    Returns a tuple of (retriever_agent, reranker_agent, generator_agent,
    condenser_agent). Use this to share agent objects across LangGraph and
    ADK orchestration.
    """
    cfg = cfg or _default_settings

    resolved_llm = llm or build_llm(cfg)
    resolved_history_llm = history_llm or build_history_llm(cfg)
    resolved_embedder = embedder or build_embedder(cfg)
    resolved_retriever = retriever or build_retriever(cfg, resolved_embedder)
    resolved_reranker = reranker or build_reranker(cfg, resolved_llm)

    retrieval_cache = (
        RetrievalCache(max_size=cfg.cache_max_size, ttl_seconds=cfg.cache_ttl_seconds)
        if cfg.cache_enabled
        else None
    )

    retriever_agent = RetrieverAgent(
        retriever=resolved_retriever,
        embedder=resolved_embedder,
        top_k=cfg.retrieval_k,
        relevance_threshold=cfg.relevance_threshold,
        cache=retrieval_cache,
        cache_strategy=cfg.retrieval_strategy,
    )
    reranker_agent = RerankerAgent(
        reranker=resolved_reranker,
        top_k=cfg.reranker_top_k,
    )
    generator_agent = GeneratorAgent(
        llm=resolved_llm,
        confidence_threshold=cfg.confidence_threshold,
    )
    condenser_agent = CondenserAgent(llm=resolved_history_llm)

    log.info(
        "factory.create_agents",
        retrieval_strategy=cfg.retrieval_strategy,
        reranker_strategy=cfg.reranker_strategy,
        retrieval_k=cfg.retrieval_k,
        reranker_top_k=cfg.reranker_top_k,
    )

    return retriever_agent, reranker_agent, generator_agent, condenser_agent


def create_librarian(
    cfg: LibrarySettings | None = None,
    *,
    llm: LLMClient | None = None,
    history_llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
    snippet_retriever: Retriever | None = None,
) -> CompiledStateGraph:
    """Build and return a compiled LibrarianGraph.

    Strategy selection follows *cfg* (defaults to module-level ``settings``).
    Any component can be overridden by passing it directly.

    Returns a LangGraph ``CompiledGraph`` ready for ``ainvoke``.
    """
    cfg = cfg or _default_settings

    from librarian.otel import setup_otel

    setup_otel()

    log.info(
        "librarian.factory.build",
        retrieval_strategy=cfg.retrieval_strategy,
        reranker_strategy=cfg.reranker_strategy,
        planning_mode=cfg.planning_mode,
        confidence_threshold=cfg.confidence_threshold,
        snippet_retriever=snippet_retriever is not None,
    )

    retriever_agent, reranker_agent, generator_agent, condenser_agent = create_agents(
        cfg, llm=llm, history_llm=history_llm, embedder=embedder,
        retriever=retriever, reranker=reranker,
    )
    resolved_checkpointer = build_checkpointer(cfg)

    return build_graph(
        retriever=retriever_agent._retriever,
        embedder=retriever_agent._embedder,
        reranker=reranker_agent._reranker,
        llm=generator_agent._llm,
        history_llm=condenser_agent._llm,
        history_condenser=condenser_agent,
        snippet_retriever=snippet_retriever,
        cache=retriever_agent._cache,
        cache_strategy=cfg.retrieval_strategy,
        retrieval_k=cfg.retrieval_k,
        reranker_top_k=cfg.reranker_top_k,
        relevance_threshold=cfg.relevance_threshold,
        confidence_threshold=cfg.confidence_threshold,
        max_crag_retries=cfg.max_crag_retries,
        max_query_variants=cfg.max_query_variants,
        checkpointer=resolved_checkpointer,
    )


def create_ingestion_pipeline(
    cfg: LibrarySettings | None = None,
    *,
    embedder: Embedder | None = None,
    retriever: Retriever | None = None,
    chunker: Chunker | None = None,
    retrieval_cache: RetrievalCache | None = None,
) -> IngestionPipeline:
    """Build an ``IngestionPipeline`` for raw-text -> vectorDB + metadataDB + traceDB.

    Returns a pipeline that can be used independently of the librarian graph.

    **Single-writer constraint (Chroma):** ``ChromaRetriever`` uses
    ``PersistentClient`` which holds a process-level write lock.  Concurrent
    within-process upserts are serialised via an ``asyncio.Lock()``, but
    multi-process ingest (e.g. parallel Fargate tasks) will fail with a lock
    error.  For multi-worker ingest, set ``retrieval_strategy=opensearch``.
    """
    from librarian.ingestion.pipeline import IngestionPipeline

    cfg = cfg or _default_settings

    log.info(
        "librarian.factory.ingestion_pipeline",
        retrieval_strategy=cfg.retrieval_strategy,
        ingestion_strategy=cfg.ingestion_strategy,
    )

    resolved_embedder = embedder or build_embedder(cfg)
    resolved_retriever = retriever or build_retriever(cfg, resolved_embedder)
    resolved_chunker = chunker or build_chunker(cfg)
    metadata_db, snippet_db = build_storage(cfg)

    return IngestionPipeline(
        chunker=resolved_chunker,
        embedder=resolved_embedder,
        vector_store=resolved_retriever,
        metadata_db=metadata_db,
        snippet_db=snippet_db,
        retrieval_cache=retrieval_cache,
    )
