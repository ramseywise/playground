"""Tests for the backend triage module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from interfaces.api import deps
from interfaces.api.app import app
from interfaces.api.triage import TriageDecision, TriageService


# ---------------------------------------------------------------------------
# Unit: TriageService.decide
# ---------------------------------------------------------------------------


class TestTriageRouting:
    """Verify that queries are routed to the correct destination."""

    def setup_method(self) -> None:
        self.svc = TriageService()

    def test_conversational_routes_direct(self) -> None:
        d = self.svc.decide("hello")
        assert d.route == "direct"
        assert d.intent == "conversational"

    def test_out_of_scope_routes_escalation(self) -> None:
        d = self.svc.decide("what's the weather today")
        assert d.route == "escalation"
        assert d.intent == "out_of_scope"

    def test_lookup_routes_librarian(self) -> None:
        d = self.svc.decide("find the API key docs")
        assert d.route == "librarian"
        assert d.intent == "lookup"

    def test_explore_routes_librarian(self) -> None:
        d = self.svc.decide("explain how the system works")
        assert d.route == "librarian"
        assert d.intent == "explore"

    def test_compare_routes_librarian(self) -> None:
        d = self.svc.decide("compare JWT vs session tokens")
        assert d.route == "librarian"
        assert d.intent == "compare"

    def test_bedrock_bypasses_classification(self) -> None:
        d = self.svc.decide("hello", backend="bedrock")
        assert d.route == "bedrock"

    def test_google_adk_bypasses_classification(self) -> None:
        d = self.svc.decide("hello", backend="google_adk")
        assert d.route == "google_adk"

    def test_adk_bedrock_bypasses_classification(self) -> None:
        d = self.svc.decide("hello", backend="adk_bedrock")
        assert d.route == "adk_bedrock"

    def test_adk_custom_rag_bypasses_classification(self) -> None:
        d = self.svc.decide("hello", backend="adk_custom_rag")
        assert d.route == "adk_custom_rag"

    def test_adk_hybrid_bypasses_classification(self) -> None:
        d = self.svc.decide("hello", backend="adk_hybrid")
        assert d.route == "adk_hybrid"

    def test_thanks_routes_direct(self) -> None:
        d = self.svc.decide("thank you for your help")
        assert d.route == "direct"
        assert d.intent == "conversational"


class TestTriageDecisionContent:
    """Verify decision payloads have the right shape."""

    def setup_method(self) -> None:
        self.svc = TriageService()

    def test_escalation_has_response(self) -> None:
        d = self.svc.decide("what's the stock price")
        assert d.response is not None
        assert len(d.response) > 0

    def test_direct_has_response(self) -> None:
        d = self.svc.decide("hey there")
        assert d.response is not None
        assert len(d.response) > 0

    def test_librarian_response_is_none(self) -> None:
        d = self.svc.decide("what is authentication?")
        assert d.response is None

    def test_bedrock_response_is_none(self) -> None:
        d = self.svc.decide("anything", backend="bedrock")
        assert d.response is None

    def test_google_adk_response_is_none(self) -> None:
        d = self.svc.decide("anything", backend="google_adk")
        assert d.response is None

    def test_confidence_in_valid_range(self) -> None:
        for query in ["hello", "weather", "find docs", "compare x vs y"]:
            d = self.svc.decide(query)
            assert 0.0 <= d.confidence <= 1.0


class TestTriageFallback:
    """Verify bedrock fallback when graph is not ready."""

    def test_fallback_to_bedrock_when_graph_not_ready(self) -> None:
        svc = TriageService(
            graph_ready=lambda: False,
            bedrock_available=lambda: True,
        )
        d = svc.decide("find auth docs")
        assert d.route == "bedrock"
        assert d.intent == "lookup"

    def test_no_fallback_when_graph_ready(self) -> None:
        svc = TriageService(
            graph_ready=lambda: True,
            bedrock_available=lambda: True,
        )
        d = svc.decide("find auth docs")
        assert d.route == "librarian"

    def test_no_fallback_when_bedrock_unavailable(self) -> None:
        svc = TriageService(
            graph_ready=lambda: False,
            bedrock_available=lambda: False,
        )
        d = svc.decide("find auth docs")
        assert d.route == "librarian"

    def test_fallback_does_not_affect_escalation(self) -> None:
        svc = TriageService(
            graph_ready=lambda: False,
            bedrock_available=lambda: True,
        )
        d = svc.decide("what's the weather")
        assert d.route == "escalation"

    def test_fallback_does_not_affect_direct(self) -> None:
        svc = TriageService(
            graph_ready=lambda: False,
            bedrock_available=lambda: True,
        )
        d = svc.decide("hello")
        assert d.route == "direct"

    def test_fallback_applies_to_explore(self) -> None:
        svc = TriageService(
            graph_ready=lambda: False,
            bedrock_available=lambda: True,
        )
        d = svc.decide("explain how auth works")
        assert d.route == "bedrock"
        assert d.intent == "explore"

    def test_fallback_applies_to_compare(self) -> None:
        svc = TriageService(
            graph_ready=lambda: False,
            bedrock_available=lambda: True,
        )
        d = svc.decide("compare JWT vs sessions")
        assert d.route == "bedrock"
        assert d.intent == "compare"

    def test_fallback_to_google_adk_when_bedrock_unavailable(self) -> None:
        svc = TriageService(
            graph_ready=lambda: False,
            bedrock_available=lambda: False,
            google_adk_available=lambda: True,
        )
        d = svc.decide("find auth docs")
        assert d.route == "google_adk"
        assert d.intent == "lookup"

    def test_bedrock_preferred_over_google_adk_for_fallback(self) -> None:
        svc = TriageService(
            graph_ready=lambda: False,
            bedrock_available=lambda: True,
            google_adk_available=lambda: True,
        )
        d = svc.decide("find auth docs")
        assert d.route == "bedrock"


class TestTriageConversationalReplies:
    """Verify the correct canned reply is selected for conversational queries."""

    def setup_method(self) -> None:
        self.svc = TriageService()

    def test_greeting_reply(self) -> None:
        d = self.svc.decide("hello")
        assert "librarian assistant" in d.response.lower()

    def test_thanks_reply(self) -> None:
        d = self.svc.decide("thanks")
        assert "welcome" in d.response.lower()

    def test_help_reply(self) -> None:
        d = self.svc.decide("help me")
        assert "search" in d.response.lower() or "corpus" in d.response.lower()


# ---------------------------------------------------------------------------
# Integration: triage via the FastAPI TestClient
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mock_graph() -> Any:
    mock = AsyncMock()
    mock.ainvoke = AsyncMock(
        return_value={
            "response": "The answer is 42.",
            "citations": [{"url": "https://example.com", "title": "Source"}],
            "confidence_score": 0.85,
            "intent": "lookup",
        }
    )

    async def astream(input_data: dict[str, Any]) -> Any:
        yield {"analyze": {"intent": "lookup"}}
        yield {"retrieve": {"retrieved_chunks": []}}
        yield {"rerank": {"reranked_chunks": [], "confidence_score": 0.85}}
        yield {"gate": {"confident": True}}
        yield {
            "generate": {
                "response": "The answer is 42.",
                "citations": [{"url": "https://example.com", "title": "Source"}],
            }
        }

    mock.astream = astream
    return mock


@pytest.fixture()
def client(_mock_graph: Any) -> TestClient:
    deps._graph = _mock_graph
    deps._pipeline = AsyncMock()
    deps._triage = TriageService()
    yield TestClient(app, raise_server_exceptions=True)
    deps._graph = None
    deps._pipeline = None
    deps._triage = None


class TestTriageViaAPI:
    def test_conversational_skips_graph(
        self, client: TestClient, _mock_graph: Any,
    ) -> None:
        resp = client.post("/api/v1/chat", json={"query": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "triage"
        assert data["intent"] == "conversational"
        _mock_graph.ainvoke.assert_not_called()

    def test_out_of_scope_skips_graph(
        self, client: TestClient, _mock_graph: Any,
    ) -> None:
        resp = client.post("/api/v1/chat", json={"query": "what's the weather"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "triage"
        assert data["intent"] == "out_of_scope"
        _mock_graph.ainvoke.assert_not_called()

    def test_lookup_invokes_graph(
        self, client: TestClient, _mock_graph: Any,
    ) -> None:
        resp = client.post("/api/v1/chat", json={"query": "What is 42?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "librarian"
        _mock_graph.ainvoke.assert_called_once()

    def test_stream_conversational_emits_done(self, client: TestClient) -> None:
        resp = client.post("/api/v1/chat/stream", json={"query": "hi there"})
        assert resp.status_code == 200
        body = resp.text
        assert "event: status" in body
        assert "event: done" in body
        assert "triage" in body
