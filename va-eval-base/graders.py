"""Baseline graders shared across all VA services."""

from __future__ import annotations


from pydantic import BaseModel, ValidationError

from .models import EvalTask, GraderResult, ServiceResponse


class _NavButton(BaseModel):
    """Schema stub for validation."""

    label: str
    route: str


class _AssistantResponse(BaseModel):
    """Schema stub for validation — matches both va-google-adk and va-langgraph."""

    message: str
    suggestions: list[str] = []
    nav_buttons: list[_NavButton] = []


class BaselineGrader:
    """Base class for all graders."""

    grader_type: str

    async def grade(self, task: EvalTask, response: ServiceResponse) -> GraderResult:
        raise NotImplementedError


class SchemaGrader(BaselineGrader):
    """Validates response conforms to AssistantResponse schema."""

    grader_type = "schema"

    async def grade(self, task: EvalTask, response: ServiceResponse) -> GraderResult:
        if response.error:
            return GraderResult(
                task_id=task.id,
                grader_type=self.grader_type,
                service=response.service,
                is_correct=False,
                score=0.0,
                reasoning=f"Service error: {response.error}",
                dimensions={"schema_valid": 0.0},
            )

        # Build an AssistantResponse-like dict from the normalized response
        candidate = {
            "message": response.message,
            "suggestions": response.suggestions,
            "nav_buttons": response.nav_buttons,
        }

        try:
            _AssistantResponse.model_validate(candidate)
            return GraderResult(
                task_id=task.id,
                grader_type=self.grader_type,
                service=response.service,
                is_correct=True,
                score=1.0,
                reasoning="Response validates against AssistantResponse schema.",
                dimensions={"schema_valid": 1.0},
            )
        except ValidationError as exc:
            return GraderResult(
                task_id=task.id,
                grader_type=self.grader_type,
                service=response.service,
                is_correct=False,
                score=0.0,
                reasoning=f"Schema validation failed: {exc.error_count()} error(s).",
                dimensions={"schema_valid": 0.0},
                details={"errors": [str(e) for e in exc.errors()]},
            )


class MessageQualityGrader(BaselineGrader):
    """Deterministic checks on message: non-empty, reasonable length, no errors."""

    grader_type = "message_quality"

    async def grade(self, task: EvalTask, response: ServiceResponse) -> GraderResult:
        if response.error:
            return GraderResult(
                task_id=task.id,
                grader_type=self.grader_type,
                service=response.service,
                is_correct=False,
                score=0.0,
                reasoning=f"Service error: {response.error}",
                dimensions={"non_empty": 0.0, "reasonable_length": 0.0},
            )

        msg = response.message
        non_empty = bool(msg and msg.strip())
        reasonable_length = 10 < len(msg) < 5000  # At least a few words, not excessive

        is_correct = non_empty and reasonable_length
        score = (float(non_empty) + float(reasonable_length)) / 2

        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            service=response.service,
            is_correct=is_correct,
            score=score,
            reasoning=(
                f"Message {'present' if non_empty else 'empty'}, "
                f"length {len(msg)} {'reasonable' if reasonable_length else 'out of range'}"
            ),
            dimensions={
                "non_empty": float(non_empty),
                "reasonable_length": float(reasonable_length),
                "length": float(len(msg)),
            },
        )


class RoutingGrader(BaselineGrader):
    """For orchestration services: check classified_intent matches expected_intent."""

    grader_type = "routing"

    async def grade(self, task: EvalTask, response: ServiceResponse) -> GraderResult:
        if task.expected_intent is None:
            return GraderResult(
                task_id=task.id,
                grader_type=self.grader_type,
                service=response.service,
                is_correct=True,
                score=1.0,
                reasoning="No expected_intent — skipped.",
            )

        if response.error or response.classified_intent is None:
            return GraderResult(
                task_id=task.id,
                grader_type=self.grader_type,
                service=response.service,
                is_correct=False,
                score=0.0,
                reasoning=f"No classified_intent — error={response.error}",
                dimensions={"match": 0.0},
                details={"expected": task.expected_intent, "actual": None},
            )

        is_correct = response.classified_intent == task.expected_intent

        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            service=response.service,
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            reasoning=f"Expected {task.expected_intent!r}, got {response.classified_intent!r}",
            dimensions={"match": float(is_correct)},
            details={
                "expected_intent": task.expected_intent,
                "classified_intent": response.classified_intent,
            },
        )
