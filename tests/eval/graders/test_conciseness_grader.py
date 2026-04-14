from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from eval.graders.conciseness_grader import ConcisenessGrader
from eval.models import EvalTask


def _make_task(response: str = "A short answer.") -> EvalTask:
    return EvalTask(
        id="t1",
        query="How do I reset my password?",
        metadata={"response": response},
    )


class TestDeterministicMode:
    @pytest.mark.asyncio()
    async def test_within_budget(self):
        grader = ConcisenessGrader(llm=None, expected_tokens=100)
        task = _make_task("This is a short answer with about ten words total.")
        result = await grader.grade(task)

        assert result.is_correct is True
        assert result.dimensions["within_budget"] == 1.0
        assert result.dimensions["token_ratio"] < 2.0
        assert result.grader_type == "conciseness"

    @pytest.mark.asyncio()
    async def test_verbose_response_fails(self):
        grader = ConcisenessGrader(llm=None, expected_tokens=10)
        long_response = " ".join(["word"] * 50)
        task = _make_task(long_response)
        result = await grader.grade(task)

        assert result.is_correct is False
        assert result.dimensions["within_budget"] == 0.0
        assert result.dimensions["token_ratio"] == 5.0

    @pytest.mark.asyncio()
    async def test_no_padding_score_without_llm(self):
        grader = ConcisenessGrader(llm=None)
        task = _make_task("Short.")
        result = await grader.grade(task)

        assert "padding_score" not in result.dimensions


class TestLLMMode:
    @pytest.mark.asyncio()
    async def test_padding_score_included(self):
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=json.dumps({
            "padding_score": 0.85,
            "is_correct": True,
            "score": 0.85,
            "reasoning": "Minor filler at start.",
        }))
        grader = ConcisenessGrader(llm=mock_llm, expected_tokens=100)
        task = _make_task("A reasonable answer.")
        result = await grader.grade(task)

        assert "padding_score" in result.dimensions
        assert result.dimensions["padding_score"] == 0.85

    @pytest.mark.asyncio()
    async def test_llm_parse_failure_returns_zero_padding(self):
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="not json")
        grader = ConcisenessGrader(llm=mock_llm, expected_tokens=100)
        task = _make_task("A reasonable answer.")
        result = await grader.grade(task)

        assert result.dimensions["padding_score"] == 0.0


class TestEmptyResponse:
    @pytest.mark.asyncio()
    async def test_empty_response_no_raise(self):
        grader = ConcisenessGrader(llm=None, expected_tokens=100)
        task = _make_task("")
        result = await grader.grade(task)

        assert result.dimensions["token_ratio"] == 0.0
        assert result.is_correct is True


class TestCustomMaxRatio:
    @pytest.mark.asyncio()
    async def test_custom_ratio_threshold(self):
        grader = ConcisenessGrader(llm=None, max_ratio=1.5, expected_tokens=10)
        task = _make_task(" ".join(["word"] * 20))
        result = await grader.grade(task)

        assert result.is_correct is False
        assert result.dimensions["token_ratio"] == 2.0
