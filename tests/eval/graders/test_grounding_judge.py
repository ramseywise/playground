from __future__ import annotations

import json

import pytest

from eval.graders.grounding_judge import GroundingJudge


class TestHappyPath:
    @pytest.mark.asyncio()
    async def test_fully_grounded(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "claims_made": 3,
            "claims_grounded": 3,
            "claims_hallucinated": 0,
            "claims_unverifiable": 0,
            "claims_parametric": 0,
            "grounding_ratio": 1.0,
            "has_hallucination": 0.0,
            "parametric_override": 0.0,
            "is_correct": True,
            "score": 1.0,
            "reasoning": "All claims supported by context.",
        })
        judge = GroundingJudge(mock_llm)
        result = await judge.grade(make_task())

        assert result.is_correct is True
        assert result.score == 1.0
        assert result.grader_type == "grounding"


class TestDimensions:
    @pytest.mark.asyncio()
    async def test_all_dimensions_populated(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "claims_made": 5,
            "claims_grounded": 4,
            "claims_hallucinated": 0,
            "claims_unverifiable": 1,
            "claims_parametric": 0,
            "grounding_ratio": 0.8,
            "has_hallucination": 0.0,
            "parametric_override": 0.0,
            "is_correct": True,
            "score": 0.8,
            "reasoning": "One unverifiable claim.",
        })
        judge = GroundingJudge(mock_llm)
        result = await judge.grade(make_task())

        expected_dims = {
            "claims_made",
            "claims_grounded",
            "grounding_ratio",
            "has_hallucination",
            "parametric_override",
        }
        assert expected_dims.issubset(set(result.dimensions.keys()))


class TestHallucinationBlocksCorrectness:
    @pytest.mark.asyncio()
    async def test_high_ratio_but_hallucinated(self, mock_llm, make_task):
        """Even with high grounding_ratio, hallucination blocks is_correct."""
        mock_llm.generate.return_value = json.dumps({
            "claims_made": 5,
            "claims_grounded": 4,
            "claims_hallucinated": 1,
            "claims_unverifiable": 0,
            "claims_parametric": 0,
            "grounding_ratio": 0.8,
            "has_hallucination": 1.0,
            "parametric_override": 0.0,
            "is_correct": False,
            "score": 0.8,
            "reasoning": "One hallucinated claim about pricing.",
        })
        judge = GroundingJudge(mock_llm)
        result = await judge.grade(make_task())

        assert result.is_correct is False
        assert result.dimensions["has_hallucination"] == 1.0


class TestParametricOverride:
    @pytest.mark.asyncio()
    async def test_at_threshold(self, mock_llm, make_task):
        """parametric_override == 0.2 is the boundary — should pass."""
        mock_llm.generate.return_value = json.dumps({
            "claims_made": 5,
            "claims_grounded": 4,
            "claims_hallucinated": 0,
            "claims_unverifiable": 0,
            "claims_parametric": 1,
            "grounding_ratio": 0.8,
            "has_hallucination": 0.0,
            "parametric_override": 0.2,
            "is_correct": True,
            "score": 0.8,
            "reasoning": "Borderline — one minor parametric claim.",
        })
        judge = GroundingJudge(mock_llm)
        result = await judge.grade(make_task())
        assert result.is_correct is True

    @pytest.mark.asyncio()
    async def test_above_threshold(self, mock_llm, make_task):
        """parametric_override > 0.2 — should fail."""
        mock_llm.generate.return_value = json.dumps({
            "claims_made": 5,
            "claims_grounded": 3,
            "claims_hallucinated": 0,
            "claims_unverifiable": 0,
            "claims_parametric": 2,
            "grounding_ratio": 0.6,
            "has_hallucination": 0.0,
            "parametric_override": 0.4,
            "is_correct": False,
            "score": 0.6,
            "reasoning": "Two claims from parametric knowledge.",
        })
        judge = GroundingJudge(mock_llm)
        result = await judge.grade(make_task())
        assert result.is_correct is False
        assert result.dimensions["parametric_override"] == 0.4


class TestParseFailure:
    @pytest.mark.asyncio()
    async def test_invalid_json(self, mock_llm, make_task):
        mock_llm.generate.return_value = "could not evaluate"
        judge = GroundingJudge(mock_llm)
        result = await judge.grade(make_task())

        assert result.is_correct is False
        assert result.score == 0.0
        assert "Failed to parse" in result.reasoning


class TestBoundary:
    @pytest.mark.asyncio()
    async def test_below_ratio_threshold(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "claims_made": 5,
            "claims_grounded": 3,
            "claims_hallucinated": 0,
            "claims_unverifiable": 2,
            "claims_parametric": 0,
            "grounding_ratio": 0.6,
            "has_hallucination": 0.0,
            "parametric_override": 0.0,
            "is_correct": False,
            "score": 0.6,
            "reasoning": "Two unverifiable claims.",
        })
        judge = GroundingJudge(mock_llm)
        result = await judge.grade(make_task())
        assert result.is_correct is False


class TestEmptyResponse:
    @pytest.mark.asyncio()
    async def test_empty_response_no_raise(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "claims_made": 0,
            "claims_grounded": 0,
            "claims_hallucinated": 0,
            "claims_unverifiable": 0,
            "claims_parametric": 0,
            "grounding_ratio": 0.0,
            "has_hallucination": 0.0,
            "parametric_override": 0.0,
            "is_correct": False,
            "score": 0.0,
            "reasoning": "No claims to verify.",
        })
        judge = GroundingJudge(mock_llm)
        task = make_task(response="")
        result = await judge.grade(task)
        assert result.score == 0.0
