from __future__ import annotations

import json

import pytest

from eval.graders.epa_judge import EPAJudge


class TestHappyPath:
    @pytest.mark.asyncio()
    async def test_high_epa(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "empathy": 0.9,
            "professionalism": 0.85,
            "actionability": 0.8,
            "epa_composite": 0.85,
            "is_correct": True,
            "score": 0.85,
            "reasoning": "Strong across all dimensions.",
        })
        judge = EPAJudge(mock_llm)
        result = await judge.grade(make_task())

        assert result.is_correct is True
        assert result.score == 0.85
        assert result.grader_type == "epa_judge"


class TestDimensions:
    @pytest.mark.asyncio()
    async def test_all_dimensions_populated(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "empathy": 0.7,
            "professionalism": 0.8,
            "actionability": 0.6,
            "epa_composite": 0.7,
            "is_correct": True,
            "score": 0.7,
            "reasoning": "Actionability is the weakest dimension.",
        })
        judge = EPAJudge(mock_llm)
        result = await judge.grade(make_task())

        expected_dims = {"empathy", "professionalism", "actionability", "epa_composite"}
        assert expected_dims.issubset(set(result.dimensions.keys()))
        assert result.dimensions["empathy"] == 0.7
        assert result.dimensions["actionability"] == 0.6


class TestParseFailure:
    @pytest.mark.asyncio()
    async def test_invalid_json(self, mock_llm, make_task):
        mock_llm.generate.return_value = "this is not json"
        judge = EPAJudge(mock_llm)
        result = await judge.grade(make_task())

        assert result.is_correct is False
        assert result.score == 0.0
        assert "Failed to parse" in result.reasoning


class TestBoundary:
    @pytest.mark.asyncio()
    async def test_below_threshold(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "empathy": 0.5,
            "professionalism": 0.6,
            "actionability": 0.5,
            "epa_composite": 0.53,
            "is_correct": False,
            "score": 0.53,
            "reasoning": "All dimensions below target.",
        })
        judge = EPAJudge(mock_llm)
        result = await judge.grade(make_task())
        assert result.is_correct is False

    @pytest.mark.asyncio()
    async def test_at_threshold(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "empathy": 0.65,
            "professionalism": 0.65,
            "actionability": 0.65,
            "epa_composite": 0.65,
            "is_correct": True,
            "score": 0.65,
            "reasoning": "Just at threshold.",
        })
        judge = EPAJudge(mock_llm)
        result = await judge.grade(make_task())
        assert result.is_correct is True


class TestEmptyResponse:
    @pytest.mark.asyncio()
    async def test_empty_response_no_raise(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "empathy": 0.0,
            "professionalism": 0.0,
            "actionability": 0.0,
            "epa_composite": 0.0,
            "is_correct": False,
            "score": 0.0,
            "reasoning": "No response to evaluate.",
        })
        judge = EPAJudge(mock_llm)
        task = make_task(response="")
        result = await judge.grade(task)
        assert result.score == 0.0
