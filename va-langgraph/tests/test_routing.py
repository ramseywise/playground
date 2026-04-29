"""Unit tests for graph routing logic — no LLM calls, no MCP connections."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from graph.builder import (
    _after_guardrail,
    _is_direct,
    _route_intent,
    _LOW_CONF_THRESHOLD,
)


def _state(**kwargs):
    return {
        "messages": [HumanMessage(content="hello")],
        "intent": "support",
        "routing_confidence": 1.0,
        "blocked": False,
        **kwargs,
    }


class TestAfterGuardrail:
    def test_blocked_routes_to_blocked(self):
        assert _after_guardrail(_state(blocked=True)) == "blocked"

    def test_not_blocked_routes_to_analyze(self):
        assert _after_guardrail(_state(blocked=False)) == "analyze"

    def test_missing_blocked_key_routes_to_analyze(self):
        state = _state()
        del state["blocked"]
        assert _after_guardrail(state) == "analyze"


class TestRouteIntent:
    @pytest.mark.parametrize(
        "intent",
        [
            "invoice",
            "quote",
            "customer",
            "product",
            "email",
            "invitation",
            "insights",
            "expense",
            "banking",
            "accounting",
            "support",
            "direct",
            "escalation",
            "memory",
        ],
    )
    def test_known_intents_route_correctly(self, intent):
        result = _route_intent(_state(intent=intent, routing_confidence=1.0))
        assert result == intent

    def test_unknown_intent_falls_back_to_support(self):
        result = _route_intent(_state(intent="unknown_domain", routing_confidence=1.0))
        assert result == "support"

    def test_low_confidence_routes_to_direct(self):
        low = _LOW_CONF_THRESHOLD - 0.01
        result = _route_intent(_state(intent="invoice", routing_confidence=low))
        assert result == "direct"

    def test_confidence_at_threshold_routes_to_intent(self):
        result = _route_intent(
            _state(intent="banking", routing_confidence=_LOW_CONF_THRESHOLD)
        )
        assert result == "banking"


class TestIsDirectEdge:
    def test_direct_intent_stays_direct(self):
        assert _is_direct(_state(intent="direct")) == "direct"

    def test_non_direct_intent_goes_to_format(self):
        assert _is_direct(_state(intent="invoice")) == "format"
