"""Unit tests for compute_safety_metrics — pure function, no LLM."""

from __future__ import annotations

import pytest

from eval.metrics.safety import compute_safety_metrics
from eval.models import EvalTask, GraderResult


def _task(expected_blocked: bool | None = None, pii: bool = False) -> EvalTask:
    return EvalTask(query="test", expected_blocked=expected_blocked, contains_pii=pii)


def _result(
    task: EvalTask, blocked: bool, pii_coverage: float | None = None
) -> GraderResult:
    task.metadata["blocked"] = blocked
    dims: dict[str, float] = {}
    if pii_coverage is not None:
        dims["pii_coverage"] = pii_coverage
    return GraderResult(
        task_id=task.id,
        grader_type="safety",
        is_correct=True,
        score=1.0,
        reasoning="",
        dimensions=dims,
    )


class TestInjectionFNR:
    def test_all_injections_blocked_fnr_zero(self):
        tasks = [_task(expected_blocked=True), _task(expected_blocked=True)]
        m = compute_safety_metrics(tasks, [_result(t, blocked=True) for t in tasks])
        assert m.injection_fnr == pytest.approx(0.0)
        assert m.n_injection_tasks == 2

    def test_no_injections_blocked_fnr_one(self):
        tasks = [_task(expected_blocked=True), _task(expected_blocked=True)]
        m = compute_safety_metrics(tasks, [_result(t, blocked=False) for t in tasks])
        assert m.injection_fnr == pytest.approx(1.0)


class TestInjectionFPR:
    def test_all_legitimate_pass_fpr_zero(self):
        tasks = [_task(expected_blocked=False), _task(expected_blocked=False)]
        m = compute_safety_metrics(tasks, [_result(t, blocked=False) for t in tasks])
        assert m.injection_fpr == pytest.approx(0.0)
        assert m.n_legitimate_tasks == 2

    def test_all_legitimate_blocked_fpr_one(self):
        t = _task(expected_blocked=False)
        m = compute_safety_metrics([t], [_result(t, blocked=True)])
        assert m.injection_fpr == pytest.approx(1.0)


class TestPiiCoverage:
    def test_full_coverage_returns_one(self):
        tasks = [_task(pii=True), _task(pii=True)]
        m = compute_safety_metrics(
            tasks, [_result(t, blocked=False, pii_coverage=1.0) for t in tasks]
        )
        assert m.pii_coverage == pytest.approx(1.0)
        assert m.n_pii_tasks == 2

    def test_partial_coverage_averaged(self):
        tasks = [_task(pii=True), _task(pii=True)]
        results = [
            _result(tasks[0], blocked=False, pii_coverage=1.0),
            _result(tasks[1], blocked=False, pii_coverage=0.5),
        ]
        m = compute_safety_metrics(tasks, results)
        assert m.pii_coverage == pytest.approx(0.75)

    def test_no_pii_tasks_coverage_defaults_to_one(self):
        t = _task(expected_blocked=True)
        m = compute_safety_metrics([t], [_result(t, blocked=True)])
        assert m.pii_coverage == pytest.approx(1.0)
        assert m.n_pii_tasks == 0


class TestEdgeCases:
    def test_empty_inputs(self):
        m = compute_safety_metrics([], [])
        assert m.injection_fnr == 0.0
        assert m.injection_fpr == 0.0
        assert m.pii_coverage == 1.0

    def test_non_safety_results_ignored(self):
        t = _task(expected_blocked=True)
        t.metadata["blocked"] = False
        r = GraderResult(
            task_id=t.id,
            grader_type="routing",
            is_correct=True,
            score=1.0,
            reasoning="",
        )
        m = compute_safety_metrics([t], [r])
        assert m.n_injection_tasks == 0
