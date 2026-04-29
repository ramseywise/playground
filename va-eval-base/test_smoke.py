"""Smoke test: validate imports and basic structure."""

from __future__ import annotations


from . import (
    EvalReport,
    EvalTask,
    GraderResult,
    MessageQualityGrader,
    RoutingGrader,
    SchemaGrader,
    ServiceResponse,
    load_clara_fixtures,
    print_report,
)


def test_imports():
    """All expected exports should import without error."""
    assert EvalTask is not None
    assert ServiceResponse is not None
    assert GraderResult is not None
    assert EvalReport is not None
    assert SchemaGrader is not None
    assert MessageQualityGrader is not None
    assert RoutingGrader is not None
    assert load_clara_fixtures is not None
    assert print_report is not None


def test_eval_task_creation():
    """EvalTask can be instantiated."""
    task = EvalTask(
        query="Test query",
        expected_intent="invoice",
        category="invoice",
    )
    assert task.query == "Test query"
    assert task.expected_intent == "invoice"
    assert task.language == "de"
    assert task.source == "clara_raw"


def test_service_response_creation():
    """ServiceResponse can be instantiated."""
    response = ServiceResponse(
        service="va-langgraph",
        task_id="test-id",
        raw_response={"message": "test"},
        message="Hello world",
        latency_ms=100.5,
    )
    assert response.service == "va-langgraph"
    assert response.message == "Hello world"
    assert response.latency_ms == 100.5
    assert response.error is None


def test_grader_result_creation():
    """GraderResult can be instantiated."""
    result = GraderResult(
        task_id="task-1",
        grader_type="schema",
        service="va-langgraph",
        is_correct=True,
        score=0.95,
        reasoning="Valid schema",
    )
    assert result.is_correct is True
    assert result.score == 0.95
    assert result.task_id == "task-1"
