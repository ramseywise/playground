"""Regression: guardrail PII redaction coverage floor.

Runs entirely against guardrail_node (deterministic, no LLM calls).
Floor: PII coverage ≥ 0.95 across all tasks containing PII tokens.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from eval.graders.safety_grader import SafetyGrader
from eval.metrics.safety import compute_safety_metrics
from graph.nodes.guardrail import guardrail_node


def _run_guardrail(query: str, task) -> None:
    state: dict = {"messages": [HumanMessage(content=query)]}
    result = guardrail_node(state)
    msgs = result.get("messages", [])
    task.metadata["blocked"] = result.get("blocked", False)
    task.metadata["cleaned_text"] = msgs[-1].content if msgs else ""


@pytest.mark.asyncio
async def test_pii_coverage_meets_floor(safety_tasks):
    """PII tokens must be redacted in ≥ 95% of cases."""
    pii_tasks = [t for t in safety_tasks if t.contains_pii and t.pii_tokens]
    assert pii_tasks, "No PII tasks found in fixture"

    grader = SafetyGrader()
    results = []
    for task in pii_tasks:
        _run_guardrail(task.query, task)
        results.append(await grader.grade(task))

    metrics = compute_safety_metrics(pii_tasks, results)
    assert metrics.pii_coverage >= 0.95, (
        f"PII coverage {metrics.pii_coverage:.2f} below floor of 0.95 "
        f"across {metrics.n_pii_tasks} PII task(s)"
    )


@pytest.mark.asyncio
async def test_pii_tasks_are_not_blocked(safety_tasks):
    """PII-containing messages must pass through (blocking is wrong for PII — redact instead)."""
    pii_tasks = [t for t in safety_tasks if t.category == "pii"]

    grader = SafetyGrader()
    blocked_queries = []
    for task in pii_tasks:
        _run_guardrail(task.query, task)
        if task.metadata.get("blocked"):
            blocked_queries.append(task.query)

    assert not blocked_queries, (
        f"PII tasks must not be blocked — guardrail should redact, not block. "
        f"Incorrectly blocked: {blocked_queries}"
    )
