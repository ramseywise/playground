"""ADK agent assembly — config-driven, using shared component builders.

Provides factory functions for all ADK-based orchestration options:
- Option 2: ``create_bedrock_agent`` — ADK + Bedrock KB (managed RAG)
- Option 3: ``create_custom_rag_agent`` — ADK + custom tools (Gemini orchestration)
- Option 4: ``create_hybrid_agent`` — ADK wrapping the full LangGraph pipeline
- Option 5: ``create_coordinator`` — Multi-agent router (Option 3 + Option 4)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    retriever_agent, reranker_agent, _, condenser_agent = create_agents(
        cfg,
        embedder=embedder,
        retriever=retriever,
        reranker=reranker,
        history_llm=condenser_llm,
    )

    log.info(
        "adk.factory.custom_rag",
        retrieval_strategy=cfg.retrieval_strategy,
        reranker_strategy=cfg.reranker_strategy,
        model=model,
    )

    return create_custom_rag_agent(
        cfg,
        retriever=retriever_agent._retriever,
        embedder=retriever_agent._embedder,
        reranker=reranker_agent._reranker,
        condenser_llm=condenser_agent._llm,
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
