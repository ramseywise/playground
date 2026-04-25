"""Unit tests for SchemaGrader — no LLM calls."""

from __future__ import annotations

import pytest

from eval.graders.schema_grader import SchemaGrader
from eval.models import EvalTask
from schema import AssistantResponse


def _task(response: dict | None) -> EvalTask:
    task = EvalTask(query="test")
    if response is not None:
        task.metadata["response"] = response
    return task


class TestValidSchema:
    @pytest.mark.asyncio
    async def test_minimal_valid_response_passes(self):
        grader = SchemaGrader()
        result = await grader.grade(_task({"message": "Hello!"}))
        assert result.is_correct is True
        assert result.score == 1.0
        assert result.dimensions["schema_valid"] == 1.0

    @pytest.mark.asyncio
    async def test_full_valid_response_passes(self):
        grader = SchemaGrader()
        response = AssistantResponse(
            message="Here are your invoices.",
            suggestions=["View details", "Export"],
            table_type="invoices",
            contact_support=False,
        ).model_dump()
        result = await grader.grade(_task(response))
        assert result.is_correct is True

    @pytest.mark.asyncio
    async def test_valid_response_with_metric_cards_passes(self):
        grader = SchemaGrader()
        response = AssistantResponse(
            message="Revenue summary.",
            metric_cards=[{"label": "Revenue", "value": "DKK 42,000"}],
        ).model_dump()
        result = await grader.grade(_task(response))
        assert result.is_correct is True


class TestInvalidSchema:
    @pytest.mark.asyncio
    async def test_missing_message_field_fails(self):
        grader = SchemaGrader()
        result = await grader.grade(_task({"suggestions": ["foo"]}))
        assert result.is_correct is False
        assert result.score == 0.0
        assert result.dimensions["schema_valid"] == 0.0
        assert result.details.get("errors")

    @pytest.mark.asyncio
    async def test_invalid_table_type_fails(self):
        grader = SchemaGrader()
        result = await grader.grade(_task({"message": "ok", "table_type": "expenses"}))
        assert result.is_correct is False

    @pytest.mark.asyncio
    async def test_none_response_fails(self):
        grader = SchemaGrader()
        result = await grader.grade(_task(None))
        assert result.is_correct is False
        assert "No response" in result.reasoning
