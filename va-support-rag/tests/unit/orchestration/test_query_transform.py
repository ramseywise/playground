"""Tests for locale-aware retrieval query expansion (no live LLM calls)."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from clients import llm
from orchestrator.langgraph import chains
from orchestrator.langgraph.nodes import retriever as query_transform
from orchestrator.langgraph.schemas import RetrievalQueryTransformOutput


@pytest.fixture(autouse=True)
def clear_caches() -> Generator[None, None, None]:
    llm.resolve_chat_model.cache_clear()
    chains.get_retrieval_query_transform_chain.cache_clear()
    yield
    llm.resolve_chat_model.cache_clear()
    chains.get_retrieval_query_transform_chain.cache_clear()


def test_expand_disabled_returns_original(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(query_transform, "RAG_RETRIEVAL_QUERY_TRANSFORM", False)
    qs, ms = query_transform.expand_queries_for_retrieval("How do I reset my password?")
    assert qs == ["How do I reset my password?"]
    assert ms == 0.0


def test_expand_skips_without_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(query_transform, "RAG_RETRIEVAL_QUERY_TRANSFORM", True)
    monkeypatch.setattr(query_transform.llm_mod, "llm_configured", lambda: False)
    qs, ms = query_transform.expand_queries_for_retrieval("Test")
    assert qs == ["Test"]
    assert ms == 0.0


def test_expand_invokes_chain_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(query_transform, "RAG_RETRIEVAL_QUERY_TRANSFORM", True)
    monkeypatch.setattr(query_transform.llm_mod, "llm_configured", lambda: True)
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = RetrievalQueryTransformOutput(
        queries=[
            "Passwort zurücksetzen",
            "Wie setze ich mein Passwort zurück",
        ]
    )
    monkeypatch.setattr(
        chains, "get_retrieval_query_transform_chain", lambda: mock_chain
    )

    qs, ms = query_transform.expand_queries_for_retrieval("reset password")
    assert qs == [
        "Passwort zurücksetzen",
        "Wie setze ich mein Passwort zurück",
    ]
    assert ms >= 0.0
    mock_chain.invoke.assert_called_once_with(
        {
            "query": "reset password",
            "target_language": "the same language as the user's question",
        }
    )


def test_expand_falls_back_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(query_transform, "RAG_RETRIEVAL_QUERY_TRANSFORM", True)
    monkeypatch.setattr(query_transform.llm_mod, "llm_configured", lambda: True)

    def _boom() -> MagicMock:
        m = MagicMock()

        def _raise(_: object) -> None:
            raise RuntimeError("structured output failed")

        m.invoke.side_effect = _raise
        return m

    monkeypatch.setattr(
        chains,
        "get_retrieval_query_transform_chain",
        _boom,
    )

    qs, _ms = query_transform.expand_queries_for_retrieval("error case")
    assert qs == ["error case"]
