"""Unit tests for graph routing — no LLMs or network."""

import pytest

from orchestrator.langgraph.routing import (
    route_after_post_answer,
    route_after_qa_policy_rerank,
    route_after_qa_policy_retrieval,
    route_after_qa_rerank_gate,
    route_after_qa_retrieval_gate,
    route_to_summarizer,
)
from orchestrator.langgraph.schemas.state import GraphState


class TestRouteAfterQaPolicyRetrieval:
    def test_rerank_branch(self) -> None:
        assert (
            route_after_qa_policy_retrieval(GraphState(qa_after_retrieval="rerank"))
            == "rerank"
        )

    def test_gate_branch(self) -> None:
        assert (
            route_after_qa_policy_retrieval(GraphState(qa_after_retrieval="gate"))
            == "gate"
        )

    def test_escalate_branch(self) -> None:
        assert (
            route_after_qa_policy_retrieval(GraphState(qa_after_retrieval="escalate"))
            == "escalate"
        )

    def test_none_defaults_to_escalate(self) -> None:
        assert route_after_qa_policy_retrieval(GraphState()) == "escalate"


class TestRouteAfterQaRetrievalGate:
    def test_retriever_action(self) -> None:
        assert (
            route_after_qa_retrieval_gate(
                GraphState(qa_retrieval_gate_action="retriever")
            )
            == "retriever"
        )

    def test_reranker_action(self) -> None:
        assert (
            route_after_qa_retrieval_gate(
                GraphState(qa_retrieval_gate_action="reranker")
            )
            == "reranker"
        )

    def test_escalation_action(self) -> None:
        assert (
            route_after_qa_retrieval_gate(
                GraphState(qa_retrieval_gate_action="escalation")
            )
            == "escalate"
        )

    def test_none_defaults_to_escalate(self) -> None:
        assert route_after_qa_retrieval_gate(GraphState()) == "escalate"


class TestRouteAfterQaPolicyRerank:
    def test_answer_branch(self) -> None:
        assert (
            route_after_qa_policy_rerank(GraphState(qa_after_rerank="answer"))
            == "answer"
        )

    def test_gate_branch(self) -> None:
        assert (
            route_after_qa_policy_rerank(GraphState(qa_after_rerank="gate")) == "gate"
        )

    def test_escalate_branch(self) -> None:
        assert (
            route_after_qa_policy_rerank(GraphState(qa_after_rerank="escalate"))
            == "escalate"
        )

    def test_none_defaults_to_escalate(self) -> None:
        assert route_after_qa_policy_rerank(GraphState()) == "escalate"


class TestRouteAfterQaRerankGate:
    def test_retriever_action(self) -> None:
        assert (
            route_after_qa_rerank_gate(GraphState(qa_rerank_gate_action="retriever"))
            == "retriever"
        )

    def test_answer_action(self) -> None:
        assert (
            route_after_qa_rerank_gate(GraphState(qa_rerank_gate_action="answer"))
            == "answer"
        )

    def test_escalation_action(self) -> None:
        assert (
            route_after_qa_rerank_gate(GraphState(qa_rerank_gate_action="escalation"))
            == "escalate"
        )

    def test_none_defaults_to_escalate(self) -> None:
        assert route_after_qa_rerank_gate(GraphState()) == "escalate"


class TestRouteToSummarizer:
    def test_always_returns_end(self) -> None:
        assert route_to_summarizer(GraphState()) == "end"


class TestRouteAfterPostAnswer:
    def test_defaults_to_end_when_evaluator_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "RAG_POST_ANSWER_EVALUATOR", False)
        assert route_after_post_answer(GraphState()) == "end"

    def test_refine_routes_to_retriever_when_evaluator_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "RAG_POST_ANSWER_EVALUATOR", True)
        assert (
            route_after_post_answer(GraphState(qa_post_answer_branch="retriever"))
            == "retriever"
        )

    def test_non_refine_routes_to_end_when_evaluator_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "RAG_POST_ANSWER_EVALUATOR", True)
        assert route_after_post_answer(GraphState()) == "end"
