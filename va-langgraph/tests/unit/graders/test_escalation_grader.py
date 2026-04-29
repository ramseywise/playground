"""Unit tests for EscalationJudge — tests _parse_result directly (no LLM calls)."""

from __future__ import annotations

from unittest.mock import MagicMock

from eval.graders.friction_grader import EscalationJudge
from eval.models import EvalTask


def _judge() -> EscalationJudge:
    return EscalationJudge(llm=MagicMock())


def _task(escalation_signal: bool = False, **kwargs) -> EvalTask:
    t = EvalTask(query="I need help.", escalation_signal=escalation_signal, **kwargs)
    t.metadata["response"] = {"message": "I'll connect you with support."}
    return t


def _parsed(warranted: float, executed: float) -> dict:
    appropriateness = 1.0 if warranted == executed else 0.0
    return {
        "escalation_warranted": warranted,
        "escalation_executed": executed,
        "appropriateness": appropriateness,
        "is_correct": appropriateness == 1.0,
        "score": appropriateness,
        "reasoning": "test",
    }


class TestEscalationJudgeParseResult:
    def test_warranted_and_executed_passes(self):
        result = _judge()._parse_result(_parsed(1.0, 1.0), _task())
        assert result.is_correct is True
        assert result.score == 1.0

    def test_warranted_not_executed_fails(self):
        result = _judge()._parse_result(_parsed(1.0, 0.0), _task())
        assert result.is_correct is False
        assert result.score == 0.0

    def test_not_warranted_but_executed_fails(self):
        result = _judge()._parse_result(_parsed(0.0, 1.0), _task())
        assert result.is_correct is False

    def test_neither_warranted_nor_executed_passes(self):
        result = _judge()._parse_result(_parsed(0.0, 0.0), _task())
        assert result.is_correct is True

    def test_dimensions_present(self):
        result = _judge()._parse_result(_parsed(1.0, 1.0), _task())
        assert "escalation_warranted" in result.dimensions
        assert "escalation_executed" in result.dimensions
        assert "appropriateness" in result.dimensions

    def test_ground_truth_match_signal_true_executed(self):
        result = _judge()._parse_result(
            _parsed(1.0, 1.0), _task(escalation_signal=True)
        )
        assert result.dimensions["ground_truth_signal"] == 1.0
        assert result.dimensions["ground_truth_match"] == 1.0

    def test_ground_truth_mismatch_signal_true_not_executed(self):
        result = _judge()._parse_result(
            _parsed(1.0, 0.0), _task(escalation_signal=True)
        )
        assert result.dimensions["ground_truth_match"] == 0.0

    def test_ground_truth_match_signal_false_not_executed(self):
        result = _judge()._parse_result(
            _parsed(0.0, 0.0), _task(escalation_signal=False)
        )
        assert result.dimensions["ground_truth_signal"] == 0.0
        assert result.dimensions["ground_truth_match"] == 1.0

    def test_grader_type(self):
        assert EscalationJudge.grader_type == "escalation_judge"
