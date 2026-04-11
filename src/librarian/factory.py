"""DI assembly — typed, config-driven dispatch.

All component construction is centralised here.  Strategy selection follows
``LibrarySettings``.  Any component can be overridden by passing it directly —
this is the primary injection point for tests and alternative configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langgraph.graph.state import CompiledStateGraph

from agents.librarian.orchestration.graph import build_graph
from agents.librarian.orchestration.history import HistoryCondenser
from agents.librarian.pipeline.ingestion.base import Chunker
from agents.librarian.pipeline.retrieval.base import Embedder, Retriever
from agents.librarian.pipeline.reranker.base import Reranker
from agents.librarian.pipeline.retrieval.cache import RetrievalCache
from agents.librarian.tools.storage.metadata_db import MetadataDB
from agents.librarian.tools.storage.snippet_db import SnippetDB
from agents.librarian.utils.config import LibrarySettings, settings as _default_settings
from agents.librarian.utils.logging import get_logger

if TYPE_CHECKING:
    from agents.librarian.tools.core.clients.llm import LLMClient
    from agents.librarian.pipeline.ingestion.pipeline import IngestionPipeline

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Builder helpers (each returns a typed component)
# ---------------------------------------------------------------------------


def _build_storage(cfg: LibrarySettings) -> tuple["MetadataDB", "SnippetDB"]:
    """Build (MetadataDB, SnippetDB) backed by cfg.duckdb_path."""
    from pathlib import Path

    from agents.librarian.tools.storage.metadata_db import MetadataDB
    from agents.librarian.tools.storage.snippet_db import SnippetDB

    db_path = cfg.duckdb_path
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    metadata_db = MetadataDB(db_path)
    snippet_db = SnippetDB(db_path)
    return cast(tuple[MetadataDB, SnippetDB], (metadata_db, snippet_db))


def _build_embedder(cfg: LibrarySettings) -> Embedder:
    from agents.librarian.pipeline.ingestion.embeddings.embedders import MultilingualEmbedder

    return MultilingualEmbedder(model_name=cfg.embedding_model)


def _build_retriever(cfg: LibrarySettings, embedder: Embedder) -> Retriever:
    if cfg.retrieval_strategy == "inmemory":
        from agents.librarian.tools.storage.vectordb.inmemory import InMemoryRetriever

        return InMemoryRetriever()

    if cfg.retrieval_strategy == "opensearch":
        from agents.librarian.tools.storage.vectordb.opensearch import OpenSearchRetriever

        return OpenSearchRetriever(index=cfg.opensearch_index)

    if cfg.retrieval_strategy == "duckdb":
        from agents.librarian.tools.storage.vectordb.duckdb import DuckDBRetriever

        return DuckDBRetriever(db_path=cfg.duckdb_path)

    # Default: chroma (persistent, no Docker required)
    from agents.librarian.tools.storage.vectordb.chroma import ChromaRetriever

    return ChromaRetriever(
        persist_dir=cfg.chroma_persist_dir,
        collection_name=cfg.chroma_collection,
    )


def _build_reranker(cfg: LibrarySettings, llm: LLMClient) -> Reranker:
    if cfg.reranker_strategy == "llm_listwise":
        from agents.librarian.pipeline.reranker.llm_listwise import LLMListwiseReranker

        return LLMListwiseReranker(llm=llm)

    if cfg.reranker_strategy == "passthrough":
        from agents.librarian.pipeline.reranker.passthrough import PassthroughReranker

        return PassthroughReranker()

    # Default: cross_encoder
    from agents.librarian.pipeline.reranker.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker()


def _build_llm(cfg: LibrarySettings) -> LLMClient:
    from agents.librarian.tools.core.clients.llm import AnthropicLLM

    return AnthropicLLM(
        model=cfg.anthropic_model_sonnet,
        api_key=cfg.anthropic_api_key,
    )


def _build_history_llm(cfg: LibrarySettings) -> LLMClient:
    from agents.librarian.tools.core.clients.llm import AnthropicLLM

    return AnthropicLLM(
        model=cfg.anthropic_model_haiku,
        api_key=cfg.anthropic_api_key,
    )


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


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

    from agents.librarian.utils.otel import setup_otel
    setup_otel()

    log.info(
        "librarian.factory.build",
        retrieval_strategy=cfg.retrieval_strategy,
        reranker_strategy=cfg.reranker_strategy,
        planning_mode=cfg.planning_mode,
        confidence_threshold=cfg.confidence_threshold,
        snippet_retriever=snippet_retriever is not None,
    )

    resolved_llm = llm or _build_llm(cfg)
    resolved_history_llm = history_llm or _build_history_llm(cfg)
    resolved_embedder = embedder or _build_embedder(cfg)
    resolved_retriever = retriever or _build_retriever(cfg, resolved_embedder)
    resolved_reranker = reranker or _build_reranker(cfg, resolved_llm)
    resolved_history_condenser = HistoryCondenser(llm=resolved_history_llm)
    retrieval_cache = (
        RetrievalCache(max_size=cfg.cache_max_size, ttl_seconds=cfg.cache_ttl_seconds)
        if cfg.cache_enabled
        else None
    )

    return build_graph(
        retriever=resolved_retriever,
        embedder=resolved_embedder,
        reranker=resolved_reranker,
        llm=resolved_llm,
        history_llm=resolved_history_llm,
        history_condenser=resolved_history_condenser,
        snippet_retriever=snippet_retriever,
        cache=retrieval_cache,
        cache_strategy=cfg.retrieval_strategy,
        retrieval_k=cfg.retrieval_k,
        reranker_top_k=cfg.reranker_top_k,
        confidence_threshold=cfg.confidence_threshold,
        max_crag_retries=cfg.max_crag_retries,
        max_query_variants=cfg.max_query_variants,
    )


def create_ingestion_pipeline(
    cfg: LibrarySettings | None = None,
    *,
    embedder: Embedder | None = None,
    retriever: Retriever | None = None,
    chunker: Chunker | None = None,
    retrieval_cache: RetrievalCache | None = None,
) -> "IngestionPipeline":
    """Build an ``IngestionPipeline`` for raw-text -> vectorDB + metadataDB + traceDB.

    Returns a pipeline that can be used independently of the librarian graph.
    """
    from agents.librarian.pipeline.ingestion.pipeline import IngestionPipeline
    from agents.librarian.pipeline.ingestion.chunking.html_aware import HtmlAwareChunker

    cfg = cfg or _default_settings

    log.info(
        "librarian.factory.ingestion_pipeline",
        retrieval_strategy=cfg.retrieval_strategy,
        ingestion_strategy=cfg.ingestion_strategy,
    )

    resolved_embedder = embedder or _build_embedder(cfg)
    resolved_retriever = retriever or _build_retriever(cfg, resolved_embedder)
    resolved_chunker = chunker or HtmlAwareChunker()
    metadata_db: MetadataDB
    snippet_db: SnippetDB
    metadata_db, snippet_db = _build_storage(cfg)

    return IngestionPipeline(
        chunker=resolved_chunker,
        embedder=resolved_embedder,
        vector_store=resolved_retriever,
        metadata_db=metadata_db,
        snippet_db=snippet_db,
        retrieval_cache=retrieval_cache,
    )
