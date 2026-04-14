"""Tests for the Librarian FastAPI API layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from interfaces.api import deps
from interfaces.api.app import app
from interfaces.api.triage import TriageService


@pytest.fixture()
def _mock_graph() -> Any:
    """Patch the graph singleton with a mock that returns a fixed result."""
    mock = AsyncMock()
    mock.ainvoke = AsyncMock(
        return_value={
            "response": "The answer is 42.",
            "citations": [{"url": "https://example.com", "title": "Source"}],
            "confidence_score": 0.85,
            "intent": "lookup",
        }
    )
    # Also mock astream to yield node updates
    mock.astream = _make_mock_astream(mock.ainvoke.return_value)
    return mock


def _make_mock_astream(final_state: dict[str, Any]) -> Any:
    """Create a mock astream that yields node-by-node updates."""

    async def astream(input_data: dict[str, Any], **kwargs: Any) -> Any:
        yield {"analyze": {"intent": "lookup"}}
        yield {"retrieve": {"retrieved_chunks": []}}
        yield {"rerank": {"reranked_chunks": [], "confidence_score": 0.85}}
        yield {"gate": {"confident": True}}
        yield {
            "generate": {
                "response": final_state["response"],
                "citations": final_state["citations"],
            }
        }

    return astream


@dataclass
class _FakeIngestionResult:
    doc_id: str = "abc123"
    chunk_count: int = 5
    snippet_count: int = 3
    skipped: bool = False


@pytest.fixture()
def _mock_pipeline() -> Any:
    """Patch the pipeline singleton with a mock."""
    mock = AsyncMock()
    mock.ingest_document = AsyncMock(return_value=_FakeIngestionResult())
    mock.ingest_s3_object = AsyncMock(return_value=_FakeIngestionResult())
    mock.ingest_s3_prefix = AsyncMock(
        return_value=[_FakeIngestionResult(), _FakeIngestionResult()]
    )
    return mock


@pytest.fixture()
def client(_mock_graph: Any, _mock_pipeline: Any) -> TestClient:
    """TestClient with graph, pipeline, and triage mocks injected."""
    deps._graph = _mock_graph
    deps._generation_sg = AsyncMock()
    deps._pipeline = _mock_pipeline
    deps._triage = TriageService()
    yield TestClient(app, raise_server_exceptions=True)
    deps._graph = None
    deps._generation_sg = None
    deps._pipeline = None
    deps._triage = None


class TestHealthEndpoint:
    def test_root_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_api_health(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestChatEndpoint:
    def test_chat_returns_response(self, client: TestClient) -> None:
        resp = client.post("/api/v1/chat", json={"query": "What is 42?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "The answer is 42."
        assert data["confidence_score"] == 0.85
        assert data["intent"] == "lookup"
        assert len(data["citations"]) == 1

    def test_chat_rejects_empty_query(self, client: TestClient) -> None:
        resp = client.post("/api/v1/chat", json={"query": ""})
        assert resp.status_code == 422

    def test_chat_requires_query(self, client: TestClient) -> None:
        resp = client.post("/api/v1/chat", json={})
        assert resp.status_code == 422


class TestStreamEndpoint:
    def test_stream_returns_sse(self, client: TestClient) -> None:
        resp = client.post("/api/v1/chat/stream", json={"query": "What is 42?"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Should contain at least status and done events
        body = resp.text
        assert "event: status" in body
        assert "event: done" in body


class TestIngestEndpoint:
    def test_ingest_inline_document(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/ingest",
            json={"document": {"text": "Hello world", "title": "Test"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["doc_id"] == "abc123"
        assert data["results"][0]["chunk_count"] == 5

    def test_ingest_s3_key(self, client: TestClient) -> None:
        resp = client.post("/api/v1/ingest", json={"s3_key": "raw/doc.md"})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1

    def test_ingest_s3_prefix(self, client: TestClient) -> None:
        resp = client.post("/api/v1/ingest", json={"s3_prefix": "raw/"})
        assert resp.status_code == 200
        # Mock returns two results for prefix
        assert len(resp.json()["results"]) == 2

    def test_ingest_requires_source(self, client: TestClient) -> None:
        resp = client.post("/api/v1/ingest", json={})
        assert resp.status_code == 422

    def test_ingest_pipeline_error(self, client: TestClient) -> None:
        deps._pipeline.ingest_document = AsyncMock(side_effect=RuntimeError("boom"))
        resp = client.post(
            "/api/v1/ingest",
            json={"document": {"text": "fail"}},
        )
        assert resp.status_code == 500


class TestEscalationSignal:
    """Verify that escalation fields are surfaced in the API response."""

    def test_chat_includes_escalate_false_when_confident(
        self, client: TestClient
    ) -> None:
        resp = client.post("/api/v1/chat", json={"query": "What is 42?"})
        assert resp.status_code == 200
        data = resp.json()
        # Mock graph returns confidence_score=0.85, confident is not set → defaults True
        assert data["confident"] is True
        assert data["escalate"] is False

    def test_chat_includes_escalate_true_when_not_confident(
        self,
        client: TestClient,
        _mock_graph: Any,
    ) -> None:
        _mock_graph.ainvoke = AsyncMock(
            return_value={
                "response": "I'm not sure.",
                "citations": [],
                "confidence_score": 0.15,
                "confident": False,
                "fallback_requested": True,
                "intent": "lookup",
            }
        )
        resp = client.post("/api/v1/chat", json={"query": "Obscure question?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["confident"] is False
        assert data["escalate"] is True
        assert data["confidence_score"] == 0.15


class TestBackendsEndpoint:
    def test_backends_returns_all_variants(self, client: TestClient) -> None:
        resp = client.get("/api/v1/backends")
        assert resp.status_code == 200
        data = resp.json()
        ids = [b["id"] for b in data["backends"]]
        assert "librarian" in ids
        assert "bedrock" in ids
        assert "google_adk" in ids
        assert "adk_bedrock" in ids
        assert "adk_custom_rag" in ids
        assert "adk_hybrid" in ids

    def test_backends_librarian_available_when_graph_ready(
        self, client: TestClient
    ) -> None:
        resp = client.get("/api/v1/backends")
        data = resp.json()
        librarian = next(b for b in data["backends"] if b["id"] == "librarian")
        assert librarian["available"] is True
        assert librarian["streaming"] is True

    def test_backends_bedrock_unavailable_when_not_configured(
        self, client: TestClient
    ) -> None:
        resp = client.get("/api/v1/backends")
        data = resp.json()
        bedrock = next(b for b in data["backends"] if b["id"] == "bedrock")
        assert bedrock["available"] is False
        assert bedrock["streaming"] is False

    def test_backends_bedrock_available_when_configured(
        self, client: TestClient
    ) -> None:
        deps._bedrock_client = AsyncMock()
        try:
            resp = client.get("/api/v1/backends")
            data = resp.json()
            bedrock = next(b for b in data["backends"] if b["id"] == "bedrock")
            assert bedrock["available"] is True
        finally:
            deps._bedrock_client = None

    def test_backends_has_labels(self, client: TestClient) -> None:
        resp = client.get("/api/v1/backends")
        data = resp.json()
        for backend in data["backends"]:
            assert "label" in backend
            assert len(backend["label"]) > 0


class TestChatBackendRouting:
    def test_chat_rejects_invalid_backend(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/chat", json={"query": "test", "backend": "invalid"}
        )
        assert resp.status_code == 422

    def test_chat_accepts_adk_bedrock_backend(self, client: TestClient) -> None:
        """adk_bedrock is accepted by Pydantic but returns 503 when not configured."""
        resp = client.post(
            "/api/v1/chat", json={"query": "test", "backend": "adk_bedrock"}
        )
        # 503 because google-adk may not be installed, or bedrock not configured
        assert resp.status_code in (200, 502, 503)

    def test_chat_accepts_adk_custom_rag_backend(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/chat", json={"query": "test", "backend": "adk_custom_rag"}
        )
        assert resp.status_code in (200, 502, 503)

    def test_chat_accepts_adk_hybrid_backend(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/chat", json={"query": "test", "backend": "adk_hybrid"}
        )
        assert resp.status_code in (200, 502, 503)

    def test_stream_rejects_adk_backends(self, client: TestClient) -> None:
        for backend in ("adk_bedrock", "adk_custom_rag", "adk_hybrid"):
            resp = client.post(
                "/api/v1/chat/stream",
                json={"query": "test", "backend": backend},
            )
            assert resp.status_code == 400
            assert "not supported" in resp.json()["detail"]


class TestLangfuseConfig:
    """Verify Langfuse config is passed to graph invocations."""

    def test_chat_passes_config_to_ainvoke(
        self,
        client: TestClient,
        _mock_graph: Any,
    ) -> None:
        resp = client.post("/api/v1/chat", json={"query": "What is 42?"})
        assert resp.status_code == 200
        _mock_graph.ainvoke.assert_called_once()
        _, kwargs = _mock_graph.ainvoke.call_args
        assert "config" in kwargs
        assert "callbacks" in kwargs["config"]

    def test_langfuse_disabled_passes_empty_callbacks(
        self,
        client: TestClient,
        _mock_graph: Any,
    ) -> None:
        """When LANGFUSE_ENABLED=false (default), callbacks list is empty."""
        resp = client.post("/api/v1/chat", json={"query": "What is 42?"})
        assert resp.status_code == 200
        _, kwargs = _mock_graph.ainvoke.call_args
        assert kwargs["config"]["callbacks"] == []
