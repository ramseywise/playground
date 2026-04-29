"""Unit tests for graph routing — no LLMs or network."""

import pytest

from orchestrator.langgraph.routing import (
    route_after_clarify,
    route_after_confirmation,
    route_after_post_answer,
    route_by_mode,
)
from orchestrator.langgraph.schemas.state import GraphState


def test_route_by_mode_qa() -> None:
    s = GraphState(mode="q&a")
    assert route_by_mode(s) == "q&a"


def test_route_by_mode_task() -> None:
    s = GraphState(mode="task_execution")
    assert route_by_mode(s) == "task_execution"


def test_route_by_mode_unknown_defaults_to_qa() -> None:
    s = GraphState(mode=None)
    assert route_by_mode(s) == "q&a"


def test_route_after_clarify_interrupt_when_missing_fields() -> None:
    s = GraphState(missing_fields=["amount"])
    assert route_after_clarify(s) == "clarify_interrupt"


def test_route_after_clarify_to_scheduler_when_complete() -> None:
    s = GraphState(missing_fields=[])
    assert route_after_clarify(s) == "scheduler"


def test_route_after_confirmation_to_answer_when_confirmed() -> None:
    s = GraphState(scheduler_confirmed=True)
    assert route_after_confirmation(s) == "answer"


def test_route_after_confirmation_back_to_scheduler_when_not_confirmed() -> None:
    s = GraphState(scheduler_confirmed=False)
    assert route_after_confirmation(s) == "scheduler"


def test_route_after_confirmation_none_goes_to_scheduler() -> None:
    s = GraphState(scheduler_confirmed=None)
    assert route_after_confirmation(s) == "scheduler"


def test_route_after_post_answer_defaults_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "RAG_POST_ANSWER_EVALUATOR", False)
    assert route_after_post_answer(GraphState()) == "end"


def test_route_after_post_answer_refine_to_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "RAG_POST_ANSWER_EVALUATOR", True)
    assert (
        route_after_post_answer(GraphState(qa_post_answer_branch="retriever"))
        == "retriever"
    )
