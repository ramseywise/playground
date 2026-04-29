"""Hybrid policy borderline helpers — no LLM."""

from orchestrator.langgraph.policies.hybrid_policy import (
    is_borderline_rerank,
    is_borderline_retrieval,
)


def test_borderline_retrieval() -> None:
    assert is_borderline_retrieval(0.35, 0.4, 0.85) is True
    assert is_borderline_retrieval(0.2, 0.4, 0.85) is False
    assert is_borderline_retrieval(0.41, 0.4, 0.85) is False


def test_borderline_rerank() -> None:
    assert is_borderline_rerank(0.22, 0.25, 0.85) is True
    assert is_borderline_rerank(0.1, 0.25, 0.85) is False
