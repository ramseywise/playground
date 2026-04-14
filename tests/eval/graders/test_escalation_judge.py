from __future__ import annotations

import json

import pytest

from eval.graders.escalation_judge import EscalationJudge


class TestHappyPath:
    @pytest.mark.asyncio()
    async def test_correct_escalation(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "escalation_warranted": 1.0,
            "escalation_executed": 1.0,
            "appropriateness": 1.0,
            "is_correct": True,
            "score": 1.0,
            "reasoning": "User requested billing dispute — escalation warranted and executed.",
        })
        judge = EscalationJudge(mock_llm)
        task = make_task(query="I want a refund for a double charge.")
        result = await judge.grade(task)

        assert result.is_correct is True
        assert result.score == 1.0
        assert result.grader_type == "escalation_judge"
        assert result.dimensions["escalation_warranted"] == 1.0
        assert result.dimensions["escalation_executed"] == 1.0
        assert result.dimensions["appropriateness"] == 1.0


class TestDimensions:
    @pytest.mark.asyncio()
    async def test_all_dimensions_populated(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "escalation_warranted": 0.0,
            "escalation_executed": 0.0,
            "appropriateness": 1.0,
            "is_correct": True,
            "score": 1.0,
            "reasoning": "In-scope question, no escalation needed or done.",
        })
        judge = EscalationJudge(mock_llm)
        result = await judge.grade(make_task())

        expected_dims = {"escalation_warranted", "escalation_executed", "appropriateness"}
        assert expected_dims.issubset(set(result.dimensions.keys()))


class TestParseFailure:
    @pytest.mark.asyncio()
    async def test_invalid_json(self, mock_llm, make_task):
        mock_llm.generate.return_value = "not valid json at all"
        judge = EscalationJudge(mock_llm)
        result = await judge.grade(make_task())

        assert result.is_correct is False
        assert result.score == 0.0
        assert "Failed to parse" in result.reasoning


class TestBoundary:
    @pytest.mark.asyncio()
    async def test_missed_escalation(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "escalation_warranted": 1.0,
            "escalation_executed": 0.0,
            "appropriateness": 0.0,
            "is_correct": False,
            "score": 0.0,
            "reasoning": "Should have escalated but did not.",
        })
        judge = EscalationJudge(mock_llm)
        result = await judge.grade(make_task())

        assert result.is_correct is False
        assert result.score == 0.0


class TestEmptyResponse:
    @pytest.mark.asyncio()
    async def test_empty_response_no_raise(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "escalation_warranted": 0.0,
            "escalation_executed": 0.0,
            "appropriateness": 1.0,
            "is_correct": True,
            "score": 1.0,
            "reasoning": "No response to evaluate.",
        })
        judge = EscalationJudge(mock_llm)
        task = make_task(response="")
        result = await judge.grade(task)

        assert result.is_correct is True
