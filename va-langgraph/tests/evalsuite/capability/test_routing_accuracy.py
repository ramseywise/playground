"""Capability: end-to-end routing accuracy via analyze_node with a real LLM.

Gated by CONFIRM_EXPENSIVE_OPS=1 — makes real LLM API calls.
Floors: overall macro-F1 ≥ 0.85, per-intent precision ≥ 0.75.
"""

from __future__ import annotations

import os

import pytest

from eval.graders.routing_grader import RoutingGrader
from eval.metrics.routing import compute_routing_metrics
from graph.nodes.analyze import analyze_node

_GATE = os.getenv("CONFIRM_EXPENSIVE_OPS") == "1"
pytestmark = pytest.mark.skipif(
    not _GATE,
    reason="Set CONFIRM_EXPENSIVE_OPS=1 to run capability tests (makes real LLM API calls)",
)


def _base_state(query: str) -> dict:
    from langchain_core.messages import HumanMessage
    return {
        "messages": [HumanMessage(content=query)],
        "session_id": "eval",
        "user_id": "eval",
        "page_url": None,
        "user_preferences": [],
        "intent": None,
        "routing_confidence": 1.0,
        "tool_results": [],
        "response": None,
        "blocked": False,
        "block_reason": None,
    }


@pytest.mark.asyncio
async def test_routing_macro_f1_meets_floor(routing_tasks):
    """Macro F1 across all 14 intents must be ≥ 0.85."""
    grader = RoutingGrader()
    results = []

    for task in routing_tasks:
        state = await analyze_node(_base_state(task.query))
        task.metadata["classified_intent"] = state.get("intent")
        task.metadata["routing_confidence"] = state.get("routing_confidence")
        results.append(await grader.grade(task))

    metrics = compute_routing_metrics(routing_tasks, results)

    print(f"\nRouting macro-F1: {metrics.overall_f1}")
    print(f"Macro precision:  {metrics.overall_precision}")
    print(f"Macro recall:     {metrics.overall_recall}")
    print("\nPer-intent breakdown:")
    for intent, m in sorted(metrics.per_intent.items()):
        print(f"  {intent:<12} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={int(m['support'])})")

    assert metrics.overall_f1 >= 0.85, (
        f"Macro F1 {metrics.overall_f1:.3f} below floor of 0.85"
    )


@pytest.mark.asyncio
async def test_per_intent_precision_meets_floor(routing_tasks):
    """Every intent with ≥ 2 samples must have precision ≥ 0.75."""
    grader = RoutingGrader()
    results = []

    for task in routing_tasks:
        state = await analyze_node(_base_state(task.query))
        task.metadata["classified_intent"] = state.get("intent")
        results.append(await grader.grade(task))

    metrics = compute_routing_metrics(routing_tasks, results)

    failing = {
        intent: m
        for intent, m in metrics.per_intent.items()
        if m["support"] >= 2 and m["precision"] < 0.75
    }

    assert not failing, (
        "Per-intent precision below 0.75 for: "
        + ", ".join(f"{k} ({v['precision']:.2f})" for k, v in failing.items())
    )
