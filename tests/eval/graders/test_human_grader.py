from __future__ import annotations

import json

import pytest

from eval.graders.human import (
    REVIEW_TAGS,
    HumanGrader,
    PendingReviewError,
    validate_tags,
)
from eval.models import EvalTask


@pytest.fixture()
def grader(tmp_path):
    return HumanGrader(review_dir=tmp_path / "reviews")


def _make_task(
    task_id: str = "t1",
    trace_id: str = "trace-abc",
    confidence: float = 0.85,
) -> EvalTask:
    return EvalTask(
        id=task_id,
        query="How do I export invoices?",
        expected_answer="Go to Invoices > Export.",
        context="Invoices can be exported from the Invoices page.",
        metadata={
            "response": "Navigate to Invoices and click Export.",
            "trace_id": trace_id,
            "confidence_score": confidence,
        },
    )


class TestSubmitWritesNewFields:
    def test_pending_contains_trace_id_and_tags(self, grader, tmp_path):
        task = _make_task()
        grader.submit(task)

        lines = (tmp_path / "reviews" / "pending.jsonl").read_text().strip().split("\n")
        entry = json.loads(lines[0])

        assert entry["trace_id"] == "trace-abc"
        assert entry["confidence_score"] == 0.85
        assert entry["tags"] == []

    def test_missing_metadata_fields_default(self, grader, tmp_path):
        task = EvalTask(id="t2", query="q", metadata={"response": "r"})
        grader.submit(task)

        lines = (tmp_path / "reviews" / "pending.jsonl").read_text().strip().split("\n")
        entry = json.loads(lines[0])

        assert entry["trace_id"] == ""
        assert entry["confidence_score"] is None
        assert entry["tags"] == []


class TestGradeReadsTags:
    @pytest.mark.asyncio()
    async def test_tags_surfaced_in_details(self, grader, tmp_path):
        task = _make_task()
        completed = {
            "task_id": "t1",
            "is_correct": True,
            "score": 0.9,
            "reasoning": "Good answer.",
            "tags": ["hallucination", "tone"],
            "trace_id": "trace-abc",
            "confidence_score": 0.85,
        }
        completed_path = tmp_path / "reviews" / "completed.jsonl"
        completed_path.write_text(json.dumps(completed) + "\n")

        result = await grader.grade(task)

        assert result.is_correct is True
        assert result.score == 0.9
        assert result.details["tags"] == ["hallucination", "tone"]
        assert result.details["trace_id"] == "trace-abc"
        assert result.details["confidence_score"] == 0.85


class TestBackwardCompatibility:
    @pytest.mark.asyncio()
    async def test_old_format_without_tags(self, grader, tmp_path):
        """Completed reviews from before the extension still parse."""
        task = _make_task()
        old_entry = {
            "task_id": "t1",
            "is_correct": True,
            "score": 0.8,
            "reasoning": "Legacy review.",
        }
        completed_path = tmp_path / "reviews" / "completed.jsonl"
        completed_path.write_text(json.dumps(old_entry) + "\n")

        result = await grader.grade(task)

        assert result.details["tags"] == []
        assert result.details["trace_id"] == ""
        assert result.details["confidence_score"] is None


class TestUnknownTagsDropped:
    @pytest.mark.asyncio()
    async def test_unknown_tag_filtered(self, grader, tmp_path):
        task = _make_task()
        completed = {
            "task_id": "t1",
            "is_correct": True,
            "score": 0.7,
            "reasoning": "Some issues.",
            "tags": ["hallucination", "xyz_invalid", "tone"],
        }
        completed_path = tmp_path / "reviews" / "completed.jsonl"
        completed_path.write_text(json.dumps(completed) + "\n")

        result = await grader.grade(task)

        assert "xyz_invalid" not in result.details["tags"]
        assert result.details["tags"] == ["hallucination", "tone"]


class TestPendingReviewError:
    @pytest.mark.asyncio()
    async def test_raises_when_not_reviewed(self, grader):
        task = _make_task()
        with pytest.raises(PendingReviewError):
            await grader.grade(task)


class TestValidateTags:
    def test_all_valid(self):
        assert validate_tags(["hallucination", "tone"]) == ["hallucination", "tone"]

    def test_strips_unknown(self):
        assert validate_tags(["hallucination", "bogus"]) == ["hallucination"]

    def test_empty_list(self):
        assert validate_tags([]) == []

    def test_review_tags_constant(self):
        assert "hallucination" in REVIEW_TAGS
        assert "retrieval_relevancy" in REVIEW_TAGS
        assert "tone" in REVIEW_TAGS
        assert "escalation_failure" in REVIEW_TAGS
        assert "context_missing" in REVIEW_TAGS
