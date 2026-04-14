from __future__ import annotations

import json

import pytest

from eval.graders.completeness_judge import CompletenessJudge


class TestHappyPath:
    @pytest.mark.asyncio()
    async def test_fully_complete(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "sub_questions_identified": 2,
            "sub_questions_answered": 2,
            "sub_question_coverage": 1.0,
            "depth_adequacy": 0.9,
            "overall_completeness": 0.95,
            "is_correct": True,
            "score": 0.95,
            "reasoning": "Both sub-questions answered with adequate depth.",
        })
        judge = CompletenessJudge(mock_llm)
        task = make_task(query="How do I reset my password and enable 2FA?")
        result = await judge.grade(task)

        assert result.is_correct is True
        assert result.score == 0.95
        assert result.grader_type == "completeness_judge"


class TestDimensions:
    @pytest.mark.asyncio()
    async def test_all_dimensions_populated(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "sub_questions_identified": 3,
            "sub_questions_answered": 2,
            "sub_question_coverage": 0.67,
            "depth_adequacy": 0.8,
            "overall_completeness": 0.73,
            "is_correct": True,
            "score": 0.73,
            "reasoning": "One sub-question about billing was not addressed.",
        })
        judge = CompletenessJudge(mock_llm)
        result = await judge.grade(make_task())

        expected_dims = {"sub_question_coverage", "depth_adequacy", "overall_completeness"}
        assert expected_dims.issubset(set(result.dimensions.keys()))


class TestParseFailure:
    @pytest.mark.asyncio()
    async def test_invalid_json(self, mock_llm, make_task):
        mock_llm.generate.return_value = "unable to parse question structure"
        judge = CompletenessJudge(mock_llm)
        result = await judge.grade(make_task())

        assert result.is_correct is False
        assert result.score == 0.0
        assert "Failed to parse" in result.reasoning


class TestBoundary:
    @pytest.mark.asyncio()
    async def test_below_threshold(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "sub_questions_identified": 3,
            "sub_questions_answered": 1,
            "sub_question_coverage": 0.33,
            "depth_adequacy": 0.5,
            "overall_completeness": 0.42,
            "is_correct": False,
            "score": 0.42,
            "reasoning": "Only one of three sub-questions addressed.",
        })
        judge = CompletenessJudge(mock_llm)
        result = await judge.grade(make_task())
        assert result.is_correct is False

    @pytest.mark.asyncio()
    async def test_at_threshold(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "sub_questions_identified": 2,
            "sub_questions_answered": 2,
            "sub_question_coverage": 1.0,
            "depth_adequacy": 0.4,
            "overall_completeness": 0.7,
            "is_correct": True,
            "score": 0.7,
            "reasoning": "Covered but shallow.",
        })
        judge = CompletenessJudge(mock_llm)
        result = await judge.grade(make_task())
        assert result.is_correct is True


class TestEmptyResponse:
    @pytest.mark.asyncio()
    async def test_empty_response_no_raise(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "sub_questions_identified": 1,
            "sub_questions_answered": 0,
            "sub_question_coverage": 0.0,
            "depth_adequacy": 0.0,
            "overall_completeness": 0.0,
            "is_correct": False,
            "score": 0.0,
            "reasoning": "No response provided.",
        })
        judge = CompletenessJudge(mock_llm)
        task = make_task(response="")
        result = await judge.grade(task)
        assert result.score == 0.0
