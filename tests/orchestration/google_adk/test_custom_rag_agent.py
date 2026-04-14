"""Tests for the ADK CustomRAGAgent with tool-based retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from librarian.config import LibrarySettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg() -> LibrarySettings:
    return LibrarySettings(
        retrieval_strategy="chroma",
        reranker_strategy="cross_encoder",
        anthropic_api_key="test",
    )


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


def test_create_custom_rag_agent_returns_agent() -> None:
    """create_custom_rag_agent should return an ADK Agent with tools wired."""
    from orchestration.google_adk.custom_rag_agent import create_custom_rag_agent

    mock_retriever = MagicMock()
    mock_embedder = MagicMock()
    mock_reranker = MagicMock()

    agent = create_custom_rag_agent(
        _cfg(),
        retriever=mock_retriever,
        embedder=mock_embedder,
        reranker=mock_reranker,
    )

    assert agent.name == "custom_rag"
    assert len(agent.tools) == 5  # analyze, condense, search, rerank, escalate
    assert "gemini-2.0-flash" in agent.model


def test_create_custom_rag_agent_custom_model() -> None:
    """Agent should accept a custom model name."""
    from orchestration.google_adk.custom_rag_agent import create_custom_rag_agent

    agent = create_custom_rag_agent(
        _cfg(),
        retriever=MagicMock(),
        embedder=MagicMock(),
        reranker=MagicMock(),
        model="gemini-2.5-flash",
    )

    assert "gemini-2.5-flash" in agent.model


def test_create_custom_rag_agent_has_instruction() -> None:
    """Agent should have retrieval instructions."""
    from orchestration.google_adk.custom_rag_agent import create_custom_rag_agent

    agent = create_custom_rag_agent(
        _cfg(),
        retriever=MagicMock(),
        embedder=MagicMock(),
        reranker=MagicMock(),
    )

    assert "search_knowledge_base" in agent.instruction
    assert "rerank_results" in agent.instruction
    assert "condense_query" in agent.instruction
    assert "analyze_query" in agent.instruction
    assert "escalate" in agent.instruction


def test_create_custom_rag_agent_configures_tools() -> None:
    """Creating the agent should call configure_tools with agent objects."""
    from orchestration.google_adk.custom_rag_agent import create_custom_rag_agent

    mock_retriever = MagicMock()
    mock_embedder = MagicMock()
    mock_reranker = MagicMock()

    with patch("orchestration.google_adk.custom_rag_agent.configure_tools") as mock_configure:
        create_custom_rag_agent(
            _cfg(),
            retriever=mock_retriever,
            embedder=mock_embedder,
            reranker=mock_reranker,
        )

        mock_configure.assert_called_once()
        call_kwargs = mock_configure.call_args.kwargs
        assert "retriever_agent" in call_kwargs
        assert "reranker_agent" in call_kwargs
        # condenser_agent should be None when no condenser_llm is provided
        assert call_kwargs["condenser_agent"] is None
