"""Unit tests for FrictionJudge — tests _parse_result directly (no LLM calls)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from eval.graders.friction_grader import FRICTION_PASS_THRESHOLD, FrictionJudge
from eval.models import EvalTask


def _judge() -> FrictionJudge:
    return FrictionJudge(llm=MagicMock())


def _task(**kwargs) -> EvalTask:
    t = EvalTask(query="How do I create an invoice?", **kwargs)
    t.metadata["response"] = {"message": "Click Invoices → New Invoice."}
    return t


class TestFrictionJudgeParseResult:
    def test_low_friction_passes(self):
        judge = _judge()
        parsed = {
            "friction_score": 0.1,
            "score": 0.9,
            "reasoning": "Direct and concise.",
        }
        result = judge._parse_result(parsed, _task())
        assert result.is_correct is True
        assert result.score == pytest.approx(0.9)
        assert result.dimensions["friction_score"] == pytest.approx(0.1)

    def test_high_friction_fails(self):
        judge = _judge()
        parsed = {
            "friction_score": 0.8,
            "score": 0.2,
            "reasoning": "Buries the answer.",
        }
        result = judge._parse_result(parsed, _task())
        assert result.is_correct is False
        assert result.score == pytest.approx(0.2)

    def test_threshold_boundary_below_passes(self):
        judge = _judge()
        parsed = {
            "friction_score": FRICTION_PASS_THRESHOLD - 0.01,
            "score": 0.66,
            "reasoning": "",
        }
        result = judge._parse_result(parsed, _task())
        assert result.is_correct is True

    def test_threshold_boundary_above_fails(self):
        judge = _judge()
        parsed = {
            "friction_score": FRICTION_PASS_THRESHOLD + 0.01,
            "score": 0.64,
            "reasoning": "",
        }
        result = judge._parse_result(parsed, _task())
        assert result.is_correct is False

    def test_dimensions_contain_friction_score(self):
        judge = _judge()
        parsed = {"friction_score": 0.3, "score": 0.7, "reasoning": "ok"}
        result = judge._parse_result(parsed, _task())
        assert "friction_score" in result.dimensions

    def test_grader_type(self):
        assert FrictionJudge.grader_type == "friction_judge"

    def test_missing_friction_score_defaults_to_high(self):
        judge = _judge()
        parsed = {"score": 0.5, "reasoning": ""}
        result = judge._parse_result(parsed, _task())
        assert result.is_correct is False

    def test_details_populated_from_task(self):
        judge = _judge()
        parsed = {"friction_score": 0.2, "score": 0.8, "reasoning": "ok"}
        task = _task(ces_rating=2, test_type="capability", expected_intent="invoice")
        result = judge._parse_result(parsed, task)
        assert result.details["ces_rating"] == 2
        assert result.details["test_type"] == "capability"
        assert result.details["intent"] == "invoice"
