"""ADK agent assembly — config-driven, using shared component builders.

Provides factory functions for all ADK-based orchestration options:
- Option 2: ``create_bedrock_agent`` — ADK + Bedrock KB (managed RAG)
- Option 3: ``create_custom_rag_agent`` — ADK + custom tools (Gemini orchestration)
- Option 4: ``create_hybrid_agent`` — ADK wrapping the full LangGraph pipeline
- Option 5: ``create_coordinator`` — Multi-agent router (Option 3 + Option 4)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from orchestration.components import (
    build_embedder,
    build_history_llm,
    build_llm,
    build_reranker,
    build_retriever,
)
from orchestration.factory import create_agents
from librarian.config import LibrarySettings, settings as _default_settings
from core.logging import get_logger

if TYPE_CHECKING:
    from google.adk.agents import Agent

    from clients.llm import LLMClient
    from librarian.retrieval.base import Embedder, Retriever
    from librarian.reranker.base import Reranker
    from orchestration.google_adk.hybrid_agent import LibrarianADKAgent

log = get_logger(__name__)


def create_custom_rag(
    cfg: LibrarySettings | None = None,
    *,
    retriever: Retriever | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    condenser_llm: LLMClient | None = None,
    model: str = "gemini-2.0-flash",
) -> Agent:
    """Build an ADK custom RAG agent with full component DI.

    Uses ``create_agents()`` to build canonical agent objects shared with
    the LangGraph pipeline, then wires them into ADK tool functions.
    """
    from orchestration.google_adk.custom_rag_agent import create_custom_rag_agent

    cfg = cfg or _default_settings

    resolved_llm = condenser_llm or build_history_llm(cfg)
    resolved_embedder = embedder or build_embedder(cfg)
    resolved_retriever = retriever or build_retriever(cfg, resolved_embedder)
    resolved_reranker = reranker or build_reranker(cfg, build_llm(cfg))

    log.info(
        "adk.factory.custom_rag",
        retrieval_strategy=cfg.retrieval_strategy,
        reranker_strategy=cfg.reranker_strategy,
        model=model,
    )

    return create_custom_rag_agent(
        cfg,
        retriever=resolved_retriever,
        embedder=resolved_embedder,
        reranker=resolved_reranker,
        condenser_llm=resolved_llm,
        model=model,
    )


def create_hybrid(
    cfg: LibrarySettings | None = None,
) -> LibrarianADKAgent:
    """Build an ADK agent wrapping the full LangGraph CRAG pipeline.

    Delegates to ``orchestration.factory.create_librarian`` for the graph,
    then wraps it in ``LibrarianADKAgent``.
    """
    from orchestration.factory import create_librarian
    from orchestration.google_adk.hybrid_agent import LibrarianADKAgent

    cfg = cfg or _default_settings
    graph = create_librarian(cfg)

    log.info(
        "adk.factory.hybrid",
        retrieval_strategy=cfg.retrieval_strategy,
        reranker_strategy=cfg.reranker_strategy,
    )

    return LibrarianADKAgent(graph=graph, cfg=cfg)


def create_coordinator_agent(
    cfg: LibrarySettings | None = None,
    *,
    model: str = "gemini-2.0-flash",
) -> Agent:
    """Build a coordinator that routes between hybrid and custom RAG agents."""
    from orchestration.google_adk.coordinator import create_coordinator

    cfg = cfg or _default_settings

    hybrid = create_hybrid(cfg)
    custom_rag = create_custom_rag(cfg, model=model)

    log.info("adk.factory.coordinator", model=model)

    return create_coordinator(hybrid, custom_rag, model=model)
