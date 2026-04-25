"""Smoke tests for graph compilation — no LLM calls, no MCP connections."""

from __future__ import annotations

from graph.builder import build_graph


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = build_graph()
    nodes = set(graph.nodes)
    expected = {
        "memory_load", "guardrail", "analyze",
        "invoice", "quote", "customer", "product",
        "email", "invitation", "insights", "expense",
        "banking", "accounting", "support",
        "direct", "escalation", "memory", "format", "blocked",
    }
    assert expected.issubset(nodes)


def test_graph_compiles_with_memory_saver():
    from langgraph.checkpoint.memory import MemorySaver
    graph = build_graph(checkpointer=MemorySaver())
    assert graph is not None
