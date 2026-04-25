"""Unit tests for RoutingGrader — no LLM calls, no graph execution."""

from __future__ import annotations

import pytest

from eval.graders.routing_grader import RoutingGrader
from eval.models import EvalTask


def _task(expected_intent: str | None, classified_intent: str | None, **kwargs) -> EvalTask:
    task = EvalTask(query="test query", expected_intent=expected_intent, **kwargs)
    if classified_intent is not None:
        task.metadata["classified_intent"] = classified_intent
    return task


class TestCorrectClassification:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("intent", [
        "invoice", "quote", "customer", "product", "email",
        "invitation", "insights", "expense", "banking",
        "accounting", "support", "direct", "escalation", "memory",
    ])
    async def test_matching_intent_passes(self, intent):
        grader = RoutingGrader()
        result = await grader.grade(_task(intent, intent))
        assert result.is_correct is True
        assert result.score == 1.0
        assert result.dimensions["match"] == 1.0

    @pytest.mark.asyncio
    async def test_mismatched_intent_fails(self):
        grader = RoutingGrader()
        result = await grader.grade(_task("invoice", "insights"))
        assert result.is_correct is False
        assert result.score == 0.0
        assert result.dimensions["match"] == 0.0

    @pytest.mark.asyncio
    async def test_reasoning_includes_expected_and_actual(self):
        grader = RoutingGrader()
        result = await grader.grade(_task("banking", "accounting"))
        assert "banking" in result.reasoning
        assert "accounting" in result.reasoning


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_no_expected_intent_skips_and_passes(self):
        grader = RoutingGrader()
        result = await grader.grade(_task(None, "invoice"))
        assert result.is_correct is True
        assert "skipped" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_missing_classified_intent_fails(self):
        task = EvalTask(query="test", expected_intent="invoice")
        grader = RoutingGrader()
        result = await grader.grade(task)
        assert result.is_correct is False

    @pytest.mark.asyncio
    async def test_details_contain_both_intents(self):
        grader = RoutingGrader()
        result = await grader.grade(_task("support", "banking"))
        assert result.details["expected_intent"] == "support"
        assert result.details["classified_intent"] == "banking"
