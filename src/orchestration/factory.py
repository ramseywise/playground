"""DI assembly — typed, config-driven dispatch.

All component construction is centralised here.  Strategy selection follows
``LibrarySettings``.  Any component can be overridden by passing it directly —
this is the primary injection point for tests and alternative configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langgraph.graph.state import CompiledStateGraph

from orchestration.langgraph.graph import build_graph
from orchestration.langgraph.history import CondenserAgent
from librarian.ingestion.base import Chunker
from librarian.retrieval.base import Embedder, Retriever
from librarian.reranker.base import Reranker
from librarian.retrieval.cache import RetrievalCache
from storage.metadata_db import MetadataDB
from storage.snippet_db import SnippetDB
from librarian.config import LibrarySettings, settings as _default_settings
from core.logging import get_logger

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from clients.llm import LLMClient
    from librarian.ingestion.pipeline import IngestionPipeline

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Builder helpers (each returns a typed component)
# ---------------------------------------------------------------------------


def _build_storage(cfg: LibrarySettings) -> tuple["MetadataDB", "SnippetDB"]:
    """Build (MetadataDB, SnippetDB) backed by cfg.duckdb_path."""
    from pathlib import Path

    from storage.metadata_db import MetadataDB
    from storage.snippet_db import SnippetDB

    db_path = cfg.duckdb_path
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    metadata_db = MetadataDB(db_path)
    snippet_db = SnippetDB(db_path)
    return cast(tuple[MetadataDB, SnippetDB], (metadata_db, snippet_db))


def _build_chunker(cfg: LibrarySettings) -> "Chunker":
    from librarian.ingestion.chunking.strategies import (
        AdjacencyChunker,
        FixedChunker,
        OverlappingChunker,
        StructuredChunker,
    )
    from librarian.ingestion.chunking.html_aware import HtmlAwareChunker
    from librarian.ingestion.chunking.parent_doc import ParentDocChunker

    dispatch = {
        "fixed": FixedChunker,
        "overlapping": OverlappingChunker,
        "structured": StructuredChunker,
        "adjacency": AdjacencyChunker,
        "parent_doc": ParentDocChunker,
        "html_aware": HtmlAwareChunker,
    }
    cls = dispatch.get(cfg.ingestion_strategy, HtmlAwareChunker)
    return cls()


def _build_embedder(cfg: LibrarySettings) -> Embedder:
    if cfg.embedding_provider == "minilm":
        from librarian.ingestion.embeddings.embedders import MiniLMEmbedder

        return MiniLMEmbedder(model_name=cfg.embedding_model)

    # Default: multilingual (intfloat/multilingual-e5-large)
    from librarian.ingestion.embeddings.embedders import MultilingualEmbedder

    return MultilingualEmbedder(model_name=cfg.embedding_model)


def warm_up_embedder(cfg: LibrarySettings | None = None) -> None:
    """Pre-load the embedding model into ``_MODEL_CACHE`` so the first real request is fast.

    Call this once during application startup (e.g. in the FastAPI lifespan).
    The model cache is process-wide, so any ``MultilingualEmbedder`` /
    ``MiniLMEmbedder`` instance created later will find the model already loaded.
    """
    cfg = cfg or _default_settings
    embedder = _build_embedder(cfg)
    embedder.embed_query("warmup")
    log.info("librarian.factory.embedder_warmup.done", model=cfg.embedding_model)


def _build_checkpointer(cfg: LibrarySettings) -> "BaseCheckpointSaver | None":
    """Build a LangGraph checkpointer based on ``cfg.checkpoint_backend``.

    - ``memory``  — in-process ``MemorySaver`` (default, no persistence across restarts)
    - ``sqlite``  — ``SqliteSaver`` backed by ``cfg.checkpoint_sqlite_path`` (local dev)
    - ``postgres`` — ``AsyncPostgresSaver`` via ``cfg.checkpoint_postgres_url`` (production)

    Returns ``None`` only when the backend is explicitly unknown so the caller
    can decide how to proceed.
    """
    from langgraph.checkpoint.base import BaseCheckpointSaver

    backend = cfg.checkpoint_backend

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        log.info("librarian.factory.checkpointer", backend="memory")
        return MemorySaver()

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            msg = (
                "langgraph-checkpoint-sqlite is required for checkpoint_backend='sqlite'. "
                "Install with: uv add langgraph-checkpoint-sqlite"
            )
            raise ImportError(msg) from exc

        from pathlib import Path

        Path(cfg.checkpoint_sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        log.info(
            "librarian.factory.checkpointer",
            backend="sqlite",
            path=cfg.checkpoint_sqlite_path,
        )
        return SqliteSaver.from_conn_string(cfg.checkpoint_sqlite_path)

    if backend == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            msg = (
                "langgraph-checkpoint-postgres is required for checkpoint_backend='postgres'. "
                "Install with: uv add langgraph-checkpoint-postgres"
            )
            raise ImportError(msg) from exc

        if not cfg.checkpoint_postgres_url:
            msg = (
                "CHECKPOINT_POSTGRES_URL must be set when checkpoint_backend='postgres'"
            )
            raise ValueError(msg)

        log.info("librarian.factory.checkpointer", backend="postgres")
        return PostgresSaver.from_conn_string(cfg.checkpoint_postgres_url)

    log.warning("librarian.factory.checkpointer.unknown", backend=backend)
    return None


def _build_retriever(cfg: LibrarySettings, embedder: Embedder) -> Retriever:
    if cfg.retrieval_strategy == "inmemory":
        from storage.vectordb.inmemory import InMemoryRetriever

        return InMemoryRetriever()

    if cfg.retrieval_strategy == "opensearch":
        from storage.vectordb.opensearch import OpenSearchRetriever

        return OpenSearchRetriever(
            index=cfg.opensearch_index,
            bm25_weight=cfg.bm25_weight,
            vector_weight=cfg.vector_weight,
        )

    if cfg.retrieval_strategy == "duckdb":
        from storage.vectordb.duckdb import DuckDBRetriever

        return DuckDBRetriever(
            db_path=cfg.duckdb_path,
            bm25_weight=cfg.bm25_weight,
            vector_weight=cfg.vector_weight,
        )

    # Default: chroma (persistent, no Docker required)
    from storage.vectordb.chroma import ChromaRetriever

    return ChromaRetriever(
        persist_dir=cfg.chroma_persist_dir,
        collection_name=cfg.chroma_collection,
        bm25_weight=cfg.bm25_weight,
        vector_weight=cfg.vector_weight,
    )


def _build_reranker(cfg: LibrarySettings, llm: LLMClient) -> Reranker:
    if cfg.reranker_strategy == "llm_listwise":
        from librarian.reranker.llm_listwise import LLMListwiseReranker

        return LLMListwiseReranker(llm=llm)

    if cfg.reranker_strategy == "passthrough":
        from librarian.reranker.passthrough import PassthroughReranker

        return PassthroughReranker()

    # Default: cross_encoder
    from librarian.reranker.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker()


def _build_llm(cfg: LibrarySettings) -> LLMClient:
    if cfg.llm_provider == "gemini":
        from clients.llm import GeminiLLM

        return GeminiLLM(model=cfg.model_gemini, api_key=cfg.gemini_api_key)

    from clients.llm import AnthropicLLM

    return AnthropicLLM(
        model=cfg.anthropic_model_sonnet,
        api_key=cfg.anthropic_api_key,
    )


def _build_history_llm(cfg: LibrarySettings) -> LLMClient:
    if cfg.llm_provider == "gemini":
        from clients.llm import GeminiLLM

        return GeminiLLM(model=cfg.model_gemini, api_key=cfg.gemini_api_key)

    from clients.llm import AnthropicLLM

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

    resolved_llm = llm or _build_llm(cfg)
    resolved_history_llm = history_llm or _build_history_llm(cfg)
    resolved_embedder = embedder or _build_embedder(cfg)
    resolved_retriever = retriever or _build_retriever(cfg, resolved_embedder)
    resolved_reranker = reranker or _build_reranker(cfg, resolved_llm)
    resolved_history_condenser = CondenserAgent(llm=resolved_history_llm)
    resolved_checkpointer = _build_checkpointer(cfg)
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
) -> "IngestionPipeline":
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

    resolved_embedder = embedder or _build_embedder(cfg)
    resolved_retriever = retriever or _build_retriever(cfg, resolved_embedder)
    resolved_chunker = chunker or _build_chunker(cfg)
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
