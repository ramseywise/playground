"""DI assembly — typed, config-driven dispatch.

All component construction is centralised here.  Strategy selection follows
``LibrarySettings``.  Any component can be overridden by passing it directly —
this is the primary injection point for tests and alternative configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents.librarian.orchestration.graph import build_graph
from agents.librarian.protocols import Chunker, Embedder, Reranker, Retriever
from agents.librarian.utils.config import LibrarySettings, settings as _default_settings
from agents.librarian.utils.logging import get_logger

if TYPE_CHECKING:
    from core.clients.llm import LLMClient

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Builder helpers (each returns a typed component)
# ---------------------------------------------------------------------------


def _build_storage(cfg: LibrarySettings) -> tuple[Any, Any]:
    """Build (MetadataDB, SnippetDB) backed by cfg.duckdb_path."""
    from pathlib import Path

    from agents.librarian.storage.metadata_db import MetadataDB
    from agents.librarian.storage.snippet_db import SnippetDB

    db_path = cfg.duckdb_path
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    return MetadataDB(db_path), SnippetDB(db_path)


def _build_embedder(cfg: LibrarySettings) -> Embedder:
    from agents.librarian.preprocessing.embedding.embedders import MultilingualEmbedder

    return MultilingualEmbedder(model_name=cfg.embedding_model)


def _build_retriever(cfg: LibrarySettings, embedder: Embedder) -> Retriever:
    if cfg.retrieval_strategy == "inmemory":
        from agents.librarian.retrieval.infra.inmemory import InMemoryRetriever

        return InMemoryRetriever()

    if cfg.retrieval_strategy == "opensearch":
        from agents.librarian.retrieval.infra.opensearch import OpenSearchRetriever

        return OpenSearchRetriever(index=cfg.opensearch_index)

    if cfg.retrieval_strategy == "duckdb":
        from agents.librarian.retrieval.infra.duckdb import DuckDBRetriever

        return DuckDBRetriever(db_path=cfg.duckdb_path)

    # Default: chroma (persistent, no Docker required)
    from agents.librarian.retrieval.infra.chroma import ChromaRetriever

    return ChromaRetriever(
        persist_dir=cfg.chroma_persist_dir,
        collection_name=cfg.chroma_collection,
    )


def _build_reranker(cfg: LibrarySettings, llm: LLMClient) -> Reranker:
    if cfg.reranker_strategy == "llm_listwise":
        from agents.librarian.reranker.llm_listwise import LLMListwiseReranker

        return LLMListwiseReranker(llm=llm)

    if cfg.reranker_strategy == "passthrough":
        from agents.librarian.reranker.passthrough import PassthroughReranker

        return PassthroughReranker()

    # Default: cross_encoder
    from agents.librarian.reranker.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker()


def _build_llm(cfg: LibrarySettings) -> LLMClient:
    from core.clients.llm import AnthropicLLM

    return AnthropicLLM(
        model=cfg.anthropic_model_sonnet,
        api_key=cfg.anthropic_api_key,
    )


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


def create_librarian(
    cfg: LibrarySettings | None = None,
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
    snippet_retriever: Retriever | None = None,
) -> Any:
    """Build and return a compiled LibrarianGraph.

    Strategy selection follows *cfg* (defaults to module-level ``settings``).
    Any component can be overridden by passing it directly.

    Returns a LangGraph ``CompiledGraph`` ready for ``ainvoke``.
    """
    cfg = cfg or _default_settings

    log.info(
        "librarian.factory.build",
        retrieval_strategy=cfg.retrieval_strategy,
        reranker_strategy=cfg.reranker_strategy,
        planning_mode=cfg.planning_mode,
        confidence_threshold=cfg.confidence_threshold,
        snippet_retriever=snippet_retriever is not None,
    )

    resolved_llm = llm or _build_llm(cfg)
    resolved_embedder = embedder or _build_embedder(cfg)
    resolved_retriever = retriever or _build_retriever(cfg, resolved_embedder)
    resolved_reranker = reranker or _build_reranker(cfg, resolved_llm)

    return build_graph(
        retriever=resolved_retriever,
        embedder=resolved_embedder,
        reranker=resolved_reranker,
        llm=resolved_llm,
        snippet_retriever=snippet_retriever,
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
) -> Any:
    """Build an ``IngestionPipeline`` for raw-text -> vectorDB + metadataDB + traceDB.

    Returns a pipeline that can be used independently of the librarian graph.
    """
    from agents.librarian.ingestion.pipeline import IngestionPipeline
    from agents.librarian.preprocessing.chunking.html_aware import HtmlAwareChunker

    cfg = cfg or _default_settings

    log.info(
        "librarian.factory.ingestion_pipeline",
        retrieval_strategy=cfg.retrieval_strategy,
        ingestion_strategy=cfg.ingestion_strategy,
    )

    resolved_embedder = embedder or _build_embedder(cfg)
    resolved_retriever = retriever or _build_retriever(cfg, resolved_embedder)
    resolved_chunker = chunker or HtmlAwareChunker()
    metadata_db, snippet_db = _build_storage(cfg)

    return IngestionPipeline(
        chunker=resolved_chunker,
        embedder=resolved_embedder,
        vector_store=resolved_retriever,
        metadata_db=metadata_db,
        snippet_db=snippet_db,
    )
