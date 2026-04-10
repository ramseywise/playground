from __future__ import annotations

from typing import Any

from agents.librarian.orchestration.graph import build_graph
from agents.librarian.utils.config import LibrarySettings, settings as _default_settings
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)


def _build_storage(cfg: LibrarySettings) -> tuple[Any, Any]:
    """Build (MetadataDB, SnippetDB) backed by cfg.duckdb_path."""
    import os
    from pathlib import Path

    from agents.librarian.storage.metadata_db import MetadataDB
    from agents.librarian.storage.snippet_db import SnippetDB

    db_path = cfg.duckdb_path
    # Ensure parent directory exists for non-in-memory paths
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    return MetadataDB(db_path), SnippetDB(db_path)


def _build_embedder(cfg: LibrarySettings) -> Any:
    from agents.librarian.preprocessing.embedding.embedders import MultilingualEmbedder

    return MultilingualEmbedder(model_name=cfg.embedding_model)


def _build_retriever(cfg: LibrarySettings, embedder: Any) -> Any:
    if cfg.retrieval_strategy == "inmemory":
        from agents.librarian.retrieval.infra.inmemory import InMemoryRetriever

        return InMemoryRetriever()

    if cfg.retrieval_strategy == "opensearch":
        from agents.librarian.retrieval.infra.opensearch import OpenSearchRetriever

        return OpenSearchRetriever(
            embedder=embedder,
            host=cfg.opensearch_url,
            index=cfg.opensearch_index,
            http_auth=(cfg.opensearch_user, cfg.opensearch_password)
            if cfg.opensearch_password
            else None,
        )

    if cfg.retrieval_strategy == "duckdb":
        from agents.librarian.retrieval.infra.duckdb import DuckDBRetriever

        return DuckDBRetriever(db_path=cfg.duckdb_path)

    # Default: chroma (persistent, no Docker required)
    from agents.librarian.retrieval.infra.chroma import ChromaRetriever

    return ChromaRetriever(
        persist_dir=cfg.chroma_persist_dir,
        collection_name=cfg.chroma_collection,
    )


def _build_reranker(cfg: LibrarySettings, llm: Any) -> Any:
    if cfg.reranker_strategy == "llm_listwise":
        from agents.librarian.reranker.llm_listwise import LLMListwiseReranker

        return LLMListwiseReranker(llm=llm)

    # Default: cross_encoder
    from agents.librarian.reranker.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker()


def _build_llm(cfg: LibrarySettings) -> Any:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=cfg.anthropic_model_sonnet,
        api_key=cfg.anthropic_api_key,  # type: ignore[arg-type]
    )


def create_librarian(
    cfg: LibrarySettings | None = None,
    *,
    # Allow injecting pre-built components (useful for tests / custom wiring)
    llm: Any = None,
    embedder: Any = None,
    retriever: Any = None,
    reranker: Any = None,
    snippet_retriever: Any = None,
) -> Any:
    """Build and return a compiled LibrarianGraph.

    Strategy selection follows ``cfg`` (defaults to module-level ``settings``).
    Any component can be overridden by passing it directly — this is the
    primary injection point for tests and alternative configurations.

    Pass *snippet_retriever* to enable the snippet path for simple factual queries.
    Use ``create_ingestion_pipeline()`` to build an ``IngestionPipeline`` that
    populates the snippet store, then wrap it with ``SnippetRetriever``.

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
    )


def create_ingestion_pipeline(
    cfg: LibrarySettings | None = None,
    *,
    embedder: Any = None,
    retriever: Any = None,
    chunker: Any = None,
) -> Any:
    """Build an ``IngestionPipeline`` for raw-text → vectorDB + metadataDB + snippetDB.

    Builds the ingestion-side components.  The returned pipeline can be used
    independently of the librarian graph.

    Example::

        from agents.librarian.factory import create_ingestion_pipeline
        from agents.librarian.storage.snippet_db import SnippetDB
        from agents.librarian.retrieval.snippet import SnippetRetriever

        pipeline = create_ingestion_pipeline()
        await pipeline.ingest_directory(Path("data/raw"))

        snippet_db = pipeline._snippet_db
        snippet_retriever = SnippetRetriever(snippet_db)
        graph = create_librarian(snippet_retriever=snippet_retriever)
    """
    from agents.librarian.ingestion.pipeline import IngestionPipeline
    from agents.librarian.preprocessing.html_aware import HtmlAwareChunker

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
