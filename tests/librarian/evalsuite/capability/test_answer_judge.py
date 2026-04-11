"""Capability tests: AnswerJudge and ClosedBookBaseline behaviour.

These test the evaluation harness itself — not the RAG pipeline.
All LLM calls are mocked; no real API calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from eval.graders.answer_eval import (
    CONFIRM_EXPENSIVE_OPS,
    AnswerJudge,
    ClosedBookBaseline,
    JudgeResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_anthropic_response(data: dict) -> MagicMock:
    content_block = MagicMock()
    content_block.text = json.dumps(data)
    resp = MagicMock()
    resp.content = [content_block]
    return resp


def _judge_with_mock(response_data: dict) -> tuple[AnswerJudge, MagicMock]:
    judge = AnswerJudge.__new__(AnswerJudge)
    judge._model = "mock-model"
    judge._max_context_chars = 3000
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response(response_data)
    judge._client = mock_client
    return judge, mock_client


# ---------------------------------------------------------------------------
# Cost gate — CONFIRM_EXPENSIVE_OPS must be False by default
# ---------------------------------------------------------------------------


def test_confirm_expensive_ops_is_false_by_default() -> None:
    """Safety: cost gate must never be committed as True."""
    assert CONFIRM_EXPENSIVE_OPS is False


def test_answer_judge_raises_without_cost_gate() -> None:
    judge = AnswerJudge.__new__(AnswerJudge)
    judge._model = "mock"
    judge._max_context_chars = 3000
    judge._client = MagicMock()
    with pytest.raises(RuntimeError, match="CONFIRM_EXPENSIVE_OPS"):
        judge.evaluate("q1", "question?", ["context"], "answer")


def test_closed_book_raises_without_cost_gate() -> None:
    baseline = ClosedBookBaseline.__new__(ClosedBookBaseline)
    baseline._model = "mock"
    baseline._client = MagicMock()
    with pytest.raises(RuntimeError, match="CONFIRM_EXPENSIVE_OPS"):
        baseline.answer("question?")


# ---------------------------------------------------------------------------
# AnswerJudge — happy path (cost gate patched)
# ---------------------------------------------------------------------------


def test_answer_judge_returns_judge_result() -> None:
    judge, _ = _judge_with_mock(
        {
            "is_correct": True,
            "score": 0.9,
            "faithfulness": 0.95,
            "relevance": 0.85,
            "completeness": 0.9,
            "reasoning": "Answer is well-grounded.",
        }
    )
    with patch(
        "eval.graders.answer_eval.CONFIRM_EXPENSIVE_OPS", True
    ):
        result = judge.evaluate(
            "q1", "what is auth?", ["auth uses API keys"], "auth uses API keys"
        )
    assert isinstance(result, JudgeResult)
    assert result.is_correct is True
    assert result.score == pytest.approx(0.9)
    assert result.query_id == "q1"


def test_answer_judge_all_score_fields_populated() -> None:
    judge, _ = _judge_with_mock(
        {
            "is_correct": False,
            "score": 0.4,
            "faithfulness": 0.3,
            "relevance": 0.5,
            "completeness": 0.4,
            "reasoning": "Partially correct.",
        }
    )
    with patch(
        "eval.graders.answer_eval.CONFIRM_EXPENSIVE_OPS", True
    ):
        result = judge.evaluate("q2", "q", ["ctx"], "ans")
    assert result.faithfulness == pytest.approx(0.3)
    assert result.relevance == pytest.approx(0.5)
    assert result.completeness == pytest.approx(0.4)


def test_answer_judge_parse_error_returns_zero_score() -> None:
    judge = AnswerJudge.__new__(AnswerJudge)
    judge._model = "mock"
    judge._max_context_chars = 3000
    mock_client = MagicMock()
    bad_response = MagicMock()
    bad_response.content = [MagicMock(text="not json at all")]
    mock_client.messages.create.return_value = bad_response
    judge._client = mock_client

    with patch(
        "eval.graders.answer_eval.CONFIRM_EXPENSIVE_OPS", True
    ):
        result = judge.evaluate("q3", "q", ["ctx"], "ans")
    assert result.is_correct is False
    assert result.score == 0.0
    assert "Parse error" in result.reasoning


def test_answer_judge_api_error_returns_zero_score() -> None:
    import anthropic

    judge = AnswerJudge.__new__(AnswerJudge)
    judge._model = "mock"
    judge._max_context_chars = 3000
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = anthropic.APIError(
        message="rate limited", request=MagicMock(), body=None
    )
    judge._client = mock_client

    with patch(
        "eval.graders.answer_eval.CONFIRM_EXPENSIVE_OPS", True
    ):
        result = judge.evaluate("q4", "q", ["ctx"], "ans")
    assert result.is_correct is False
    assert result.score == 0.0


def test_answer_judge_truncates_long_context() -> None:
    judge, mock_client = _judge_with_mock(
        {
            "is_correct": True,
            "score": 1.0,
            "faithfulness": 1.0,
            "relevance": 1.0,
            "completeness": 1.0,
            "reasoning": "ok",
        }
    )
    judge._max_context_chars = 50

    with patch(
        "eval.graders.answer_eval.CONFIRM_EXPENSIVE_OPS", True
    ):
        judge.evaluate("q5", "q", ["x" * 200], "ans")

    call_kwargs = mock_client.messages.create.call_args[1]
    user_content = call_kwargs["messages"][0]["content"]
    assert "[truncated]" in user_content


# ---------------------------------------------------------------------------
# AnswerJudge.evaluate_batch
# ---------------------------------------------------------------------------


def test_evaluate_batch_returns_all_results() -> None:
    judge, _ = _judge_with_mock(
        {
            "is_correct": True,
            "score": 0.8,
            "faithfulness": 0.8,
            "relevance": 0.8,
            "completeness": 0.8,
            "reasoning": "ok",
        }
    )
    samples = [
        {
            "query_id": f"q{i}",
            "question": "q",
            "context_chunks": ["ctx"],
            "answer": "ans",
        }
        for i in range(3)
    ]
    with patch(
        "eval.graders.answer_eval.CONFIRM_EXPENSIVE_OPS", True
    ):
        results = judge.evaluate_batch(samples)
    assert len(results) == 3
    assert all(isinstance(r, JudgeResult) for r in results)


# ---------------------------------------------------------------------------
# JudgeResult dataclass
# ---------------------------------------------------------------------------


def test_judge_result_fields() -> None:
    r = JudgeResult(
        is_correct=True,
        score=0.9,
        faithfulness=0.9,
        relevance=0.9,
        completeness=0.9,
        reasoning="good",
        query_id="q1",
    )
    assert r.query_id == "q1"
    assert r.score == 0.9
