"""Smoke tests for LangGraphRuntime — no live LLM or vector DB."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.langgraph.runtime import LangGraphRuntime
from orchestrator.runtime_protocol import AgentRuntime
from orchestrator.schemas import AgentInput, AgentOutput


def test_langgraph_runtime_satisfies_protocol() -> None:
    assert isinstance(LangGraphRuntime(), AgentRuntime)


def test_agent_input_schema() -> None:
    inp = AgentInput(query="hello", thread_id="t1")
    assert inp.query == "hello"
    assert inp.thread_id == "t1"
    assert inp.locale is None


def test_agent_output_schema() -> None:
    out = AgentOutput(answer="ok")
    assert out.escalated is False
    assert out.citations == []


@pytest.mark.asyncio
async def test_run_returns_agent_output() -> None:
    mock_result = {
        "final_answer": "Here is your answer.",
        "citations": [],
        "mode": "q&a",
        "latency_ms": {"llm_ms": 100.0},
        "qa_outcome": "answer",
    }
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=mock_result)
    runtime = LangGraphRuntime(graph=mock_graph)
    result = await runtime.run(AgentInput(query="What is X?", thread_id="t1"))

    assert isinstance(result, AgentOutput)
    assert result.answer == "Here is your answer."
    assert result.escalated is False
    assert result.pipeline_error is False
    assert result.mode == "q&a"


@pytest.mark.asyncio
async def test_run_escalated_answer() -> None:
    mock_result = {
        "final_answer": "Please contact support.",
        "citations": [],
        "mode": "q&a",
        "latency_ms": {},
        "qa_outcome": "escalate",
    }
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=mock_result)
    runtime = LangGraphRuntime(graph=mock_graph)
    result = await runtime.run(AgentInput(query="help", thread_id="t2"))

    assert result.escalated is True


@pytest.mark.asyncio
async def test_run_graph_exception_returns_pipeline_failure() -> None:
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
    runtime = LangGraphRuntime(graph=mock_graph)
    result = await runtime.run(AgentInput(query="q", thread_id="t-fail"))

    assert result.pipeline_error is True
    assert result.escalated is False
    assert "technical issue" in result.answer
