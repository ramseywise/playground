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


@pytest.fixture(scope="session")
def sevdesk_tasks() -> list[EvalTask]:
    """Real sevdesk tickets (GDPR-scrubbed). Empty list if fixture not yet generated.

    Generate with:
        cd va-langgraph
        uv run python eval/ingest/sevdesk_ingest.py
    Then run the LLM review pass per .claude/skills/gdpr-scrub/SKILL.md.
    """
    fixture_path = FIXTURES / "sevdesk_tickets.json"
    if not fixture_path.exists():
        pytest.skip("sevdesk_tickets.json not found — run eval/ingest/sevdesk_ingest.py first")
        return []
    data = json.loads(fixture_path.read_text())
    return [EvalTask(**d) for d in data]


@pytest.fixture(scope="session")
def sevdesk_capability_tasks(sevdesk_tasks: list[EvalTask]) -> list[EvalTask]:
    return [t for t in sevdesk_tasks if t.test_type == "capability"]


@pytest.fixture(scope="session")
def sevdesk_regression_tasks(sevdesk_tasks: list[EvalTask]) -> list[EvalTask]:
    return [t for t in sevdesk_tasks if t.test_type == "regression"]


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
