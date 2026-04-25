"""Deterministic schema grader: validates response against AssistantResponse."""

from __future__ import annotations

from pydantic import ValidationError

from schema import AssistantResponse
from ..models import EvalTask, GraderResult


class SchemaGrader:
    """Validates task.metadata['response'] against the AssistantResponse Pydantic schema."""

    grader_type = "schema"

    async def grade(self, task: EvalTask) -> GraderResult:
        response_dict = task.metadata.get("response")

        if response_dict is None:
            return GraderResult(
                task_id=task.id,
                grader_type=self.grader_type,
                is_correct=False,
                score=0.0,
                reasoning="No response found in task metadata.",
                dimensions={"schema_valid": 0.0},
            )

        try:
            AssistantResponse.model_validate(response_dict)
            return GraderResult(
                task_id=task.id,
                grader_type=self.grader_type,
                is_correct=True,
                score=1.0,
                reasoning="Response validates against AssistantResponse schema.",
                dimensions={"schema_valid": 1.0},
            )
        except ValidationError as exc:
            return GraderResult(
                task_id=task.id,
                grader_type=self.grader_type,
                is_correct=False,
                score=0.0,
                reasoning=f"Schema validation failed: {exc.error_count()} error(s).",
                dimensions={"schema_valid": 0.0},
                details={"errors": exc.errors()},
            )
