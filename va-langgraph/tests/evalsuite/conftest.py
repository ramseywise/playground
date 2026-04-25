"""Shared fixtures for the VA LangGraph eval suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from eval.models import EvalTask
from graph.nodes.guardrail import guardrail_node

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def routing_tasks() -> list[EvalTask]:
    data = json.loads((FIXTURES / "routing_tasks.json").read_text())
    return [EvalTask(**d) for d in data]


@pytest.fixture(scope="session")
def safety_tasks() -> list[EvalTask]:
    data = json.loads((FIXTURES / "safety_tasks.json").read_text())
    return [EvalTask(**d) for d in data]


def run_guardrail(query: str) -> dict:
    """Run guardrail_node and return normalized result for test assertions."""
    state: dict = {"messages": [HumanMessage(content=query)]}
    result = guardrail_node(state)
    msgs = result.get("messages", [])
    cleaned = msgs[-1].content if msgs else ""
    return {
        "blocked": result.get("blocked", False),
        "block_reason": result.get("block_reason"),
        "cleaned_text": cleaned,
    }
