from __future__ import annotations

import json

import pytest

from eval.graders.knowledge_override_judge import KnowledgeOverrideJudge


class TestHappyPath:
    @pytest.mark.asyncio()
    async def test_no_override(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "claims_inspected": 3,
            "parametric_claims": 0,
            "context_used": 1.0,
            "parametric_override": 0.0,
            "override_score": 1.0,
            "is_correct": True,
            "score": 1.0,
            "reasoning": "All claims are grounded in the context.",
        })
        judge = KnowledgeOverrideJudge(mock_llm)
        result = await judge.grade(make_task())

        assert result.is_correct is True
        assert result.score == 1.0
        assert result.grader_type == "knowledge_override_judge"


class TestDimensions:
    @pytest.mark.asyncio()
    async def test_all_dimensions_populated(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "claims_inspected": 4,
            "parametric_claims": 1,
            "context_used": 0.75,
            "parametric_override": 0.25,
            "override_score": 0.75,
            "is_correct": False,
            "score": 0.75,
            "reasoning": "One claim sourced from parametric knowledge.",
        })
        judge = KnowledgeOverrideJudge(mock_llm)
        result = await judge.grade(make_task())

        expected_dims = {"context_used", "parametric_override", "override_score"}
        assert expected_dims.issubset(set(result.dimensions.keys()))
        assert result.dimensions["parametric_override"] == 0.25


class TestParseFailure:
    @pytest.mark.asyncio()
    async def test_invalid_json(self, mock_llm, make_task):
        mock_llm.generate.return_value = "I cannot evaluate this."
        judge = KnowledgeOverrideJudge(mock_llm)
        result = await judge.grade(make_task())

        assert result.is_correct is False
        assert result.score == 0.0
        assert "Failed to parse" in result.reasoning


class TestBoundary:
    @pytest.mark.asyncio()
    async def test_at_threshold(self, mock_llm, make_task):
        """parametric_override == 0.2 is the boundary — should be correct."""
        mock_llm.generate.return_value = json.dumps({
            "claims_inspected": 5,
            "parametric_claims": 1,
            "context_used": 0.8,
            "parametric_override": 0.2,
            "override_score": 0.8,
            "is_correct": True,
            "score": 0.8,
            "reasoning": "Borderline — one minor parametric claim.",
        })
        judge = KnowledgeOverrideJudge(mock_llm)
        result = await judge.grade(make_task())
        assert result.is_correct is True

    @pytest.mark.asyncio()
    async def test_above_threshold(self, mock_llm, make_task):
        """parametric_override > 0.2 — should fail."""
        mock_llm.generate.return_value = json.dumps({
            "claims_inspected": 5,
            "parametric_claims": 2,
            "context_used": 0.6,
            "parametric_override": 0.4,
            "override_score": 0.6,
            "is_correct": False,
            "score": 0.6,
            "reasoning": "Two claims from parametric knowledge.",
        })
        judge = KnowledgeOverrideJudge(mock_llm)
        result = await judge.grade(make_task())
        assert result.is_correct is False


class TestEmptyResponse:
    @pytest.mark.asyncio()
    async def test_empty_response_no_raise(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "claims_inspected": 0,
            "parametric_claims": 0,
            "context_used": 0.0,
            "parametric_override": 0.0,
            "override_score": 1.0,
            "is_correct": True,
            "score": 1.0,
            "reasoning": "No claims to evaluate.",
        })
        judge = KnowledgeOverrideJudge(mock_llm)
        task = make_task(response="")
        result = await judge.grade(task)
        assert result.score == 1.0
