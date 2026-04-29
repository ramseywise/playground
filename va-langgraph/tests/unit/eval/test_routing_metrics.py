"""Unit tests for compute_routing_metrics — pure function, no LLM."""

from __future__ import annotations

import pytest

from eval.metrics.routing import compute_routing_metrics
from eval.models import EvalTask, GraderResult


def _task(expected_intent: str, classified_intent: str | None = None) -> EvalTask:
    t = EvalTask(query="q", expected_intent=expected_intent)
    if classified_intent is not None:
        t.metadata["classified_intent"] = classified_intent
    return t


def _result(task: EvalTask) -> GraderResult:
    return GraderResult(
        task_id=task.id,
        grader_type="routing",
        is_correct=True,
        score=1.0,
        reasoning="",
    )


class TestPerfectAccuracy:
    def test_all_correct_returns_f1_1(self):
        tasks = [_task("invoice", "invoice"), _task("quote", "quote")]
        m = compute_routing_metrics(tasks, [_result(t) for t in tasks])
        assert m.overall_f1 == pytest.approx(1.0)
        assert m.overall_precision == pytest.approx(1.0)
        assert m.overall_recall == pytest.approx(1.0)

    def test_per_intent_populated(self):
        tasks = [_task("invoice", "invoice")]
        m = compute_routing_metrics(tasks, [_result(tasks[0])])
        assert "invoice" in m.per_intent
        assert m.per_intent["invoice"]["f1"] == pytest.approx(1.0)
        assert m.per_intent["invoice"]["support"] == 1.0


class TestMixedAccuracy:
    def test_all_wrong_returns_zero_f1(self):
        tasks = [_task("invoice", "quote"), _task("quote", "invoice")]
        m = compute_routing_metrics(tasks, [_result(t) for t in tasks])
        assert m.overall_f1 == pytest.approx(0.0)

    def test_partial_accuracy_f1_between_zero_and_one(self):
        tasks = [_task("invoice", "invoice"), _task("invoice", "quote")]
        m = compute_routing_metrics(tasks, [_result(t) for t in tasks])
        assert 0.0 < m.overall_f1 < 1.0


class TestEdgeCases:
    def test_empty_inputs_returns_zeros(self):
        m = compute_routing_metrics([], [])
        assert m.overall_f1 == 0.0
        assert m.per_intent == {}

    def test_non_routing_results_are_ignored(self):
        t = _task("invoice", "invoice")
        r = GraderResult(
            task_id=t.id, grader_type="schema", is_correct=True, score=1.0, reasoning=""
        )
        m = compute_routing_metrics([t], [r])
        assert m.overall_f1 == 0.0

    def test_task_without_expected_intent_is_skipped(self):
        t = EvalTask(query="q")
        r = GraderResult(
            task_id=t.id,
            grader_type="routing",
            is_correct=True,
            score=1.0,
            reasoning="",
        )
        m = compute_routing_metrics([t], [r])
        assert m.overall_f1 == 0.0
