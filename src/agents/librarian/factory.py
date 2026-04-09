from __future__ import annotations

from typing import Any

from agents.librarian.orchestration.graph import build_graph
from agents.librarian.utils.config import LibrarySettings, settings as _default_settings
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)


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
) -> Any:
    """Build and return a compiled LibrarianGraph.

    Strategy selection follows ``cfg`` (defaults to module-level ``settings``).
    Any component can be overridden by passing it directly — this is the
    primary injection point for tests and alternative configurations.

    Returns a LangGraph ``CompiledGraph`` ready for ``ainvoke``.
    """
    cfg = cfg or _default_settings

    log.info(
        "librarian.factory.build",
        retrieval_strategy=cfg.retrieval_strategy,
        reranker_strategy=cfg.reranker_strategy,
        planning_mode=cfg.planning_mode,
        confidence_threshold=cfg.confidence_threshold,
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
        retrieval_k=cfg.retrieval_k,
        reranker_top_k=cfg.reranker_top_k,
        confidence_threshold=cfg.confidence_threshold,
        max_crag_retries=cfg.max_crag_retries,
    )
