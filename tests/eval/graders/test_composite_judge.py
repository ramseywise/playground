from __future__ import annotations

import json

import pytest

from eval.graders.composite_judge import CompositeJudge


class TestHappyPath:
    @pytest.mark.asyncio()
    async def test_all_metrics_pass(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "grounding": {
                "claims_made": 3,
                "claims_grounded": 3,
                "grounding_ratio": 1.0,
                "has_hallucination": 0.0,
                "parametric_override": 0.0,
                "is_correct": True,
                "score": 1.0,
                "reasoning": "All grounded.",
            },
            "epa": {
                "empathy": 0.8,
                "professionalism": 0.9,
                "actionability": 0.7,
                "epa_composite": 0.8,
                "is_correct": True,
                "score": 0.8,
                "reasoning": "Good tone.",
            },
        })
        judge = CompositeJudge(mock_llm, metrics=["grounding", "epa"])
        result = await judge.grade(make_task())

        assert result.is_correct is True
        assert result.score == pytest.approx(0.9)  # mean(1.0, 0.8)
        assert result.grader_type == "composite:epa+grounding"


class TestAllMetricsMustPass:
    @pytest.mark.asyncio()
    async def test_one_metric_fails(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "grounding": {
                "grounding_ratio": 0.5,
                "has_hallucination": 1.0,
                "parametric_override": 0.0,
                "is_correct": False,
                "score": 0.5,
                "reasoning": "Hallucinated.",
            },
            "epa": {
                "epa_composite": 0.8,
                "is_correct": True,
                "score": 0.8,
                "reasoning": "Good.",
            },
        })
        judge = CompositeJudge(mock_llm, metrics=["grounding", "epa"])
        result = await judge.grade(make_task())

        assert result.is_correct is False
        assert "grounding.grounding_ratio" in result.dimensions
        assert "epa.epa_composite" in result.dimensions


class TestPartialFailure:
    @pytest.mark.asyncio()
    async def test_missing_metric_in_json(self, mock_llm, make_task):
        """A missing sub-object should cause that metric to fail."""
        mock_llm.generate.return_value = json.dumps({
            "epa": {
                "epa_composite": 0.8,
                "is_correct": True,
                "score": 0.8,
                "reasoning": "Good.",
            },
            # "grounding" sub-object is missing
        })
        judge = CompositeJudge(mock_llm, metrics=["grounding", "epa"])
        result = await judge.grade(make_task())

        assert result.is_correct is False  # grounding fails (empty dict)
        assert result.dimensions.get("epa.epa_composite") == 0.8


class TestParseFailure:
    @pytest.mark.asyncio()
    async def test_invalid_json(self, mock_llm, make_task):
        mock_llm.generate.return_value = "not json at all"
        judge = CompositeJudge(mock_llm, metrics=["grounding"])
        result = await judge.grade(make_task())

        assert result.is_correct is False
        assert result.score == 0.0
        assert "Failed to parse" in result.reasoning


class TestGraderType:
    def test_sorted_regardless_of_input_order(self, mock_llm):
        j1 = CompositeJudge(mock_llm, metrics=["epa", "grounding", "completeness"])
        j2 = CompositeJudge(mock_llm, metrics=["grounding", "completeness", "epa"])
        assert j1.grader_type == j2.grader_type
        assert j1.grader_type == "composite:completeness+epa+grounding"


class TestSingleMetric:
    @pytest.mark.asyncio()
    async def test_single_metric_works(self, mock_llm, make_task):
        mock_llm.generate.return_value = json.dumps({
            "completeness": {
                "sub_questions_identified": 2,
                "sub_questions_answered": 2,
                "overall_completeness": 0.9,
                "is_correct": True,
                "score": 0.9,
                "reasoning": "All covered.",
            },
        })
        judge = CompositeJudge(mock_llm, metrics=["completeness"])
        result = await judge.grade(make_task())

        assert result.is_correct is True
        assert result.score == pytest.approx(0.9)
        assert result.grader_type == "composite:completeness"


class TestValidation:
    def test_unknown_metric_raises(self, mock_llm):
        with pytest.raises(ValueError, match="Unknown metrics"):
            CompositeJudge(mock_llm, metrics=["nonexistent"])

    def test_empty_metrics_raises(self, mock_llm):
        with pytest.raises(ValueError, match="At least one metric"):
            CompositeJudge(mock_llm, metrics=[])


class TestUserMessage:
    def test_context_included_when_grounding_selected(self, mock_llm, make_task):
        judge = CompositeJudge(mock_llm, metrics=["grounding"])
        msg = judge._format_user_message(
            query="q", response="r", context="ctx", expected=""
        )
        assert "Retrieved context passages:" in msg
        assert "ctx" in msg

    def test_context_omitted_when_only_epa(self, mock_llm, make_task):
        judge = CompositeJudge(mock_llm, metrics=["epa"])
        msg = judge._format_user_message(
            query="q", response="r", context="ctx", expected=""
        )
        assert "Retrieved context" not in msg
        assert "Agent response:" in msg

    def test_expected_included_when_completeness_selected(self, mock_llm, make_task):
        judge = CompositeJudge(mock_llm, metrics=["completeness"])
        msg = judge._format_user_message(
            query="q", response="r", context="", expected="exp"
        )
        assert "Expected answer (reference):" in msg
