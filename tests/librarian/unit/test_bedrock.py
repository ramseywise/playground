"""Tests for the Bedrock Knowledge Bases client and API integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from librarian.bedrock.client import (
    BedrockKBClient,
    BedrockKBResponse,
    _extract_citations,
)
from interfaces.api import deps
from interfaces.api.app import app
from librarian.config import LibrarySettings


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------


class TestExtractCitations:
    def test_empty_list(self) -> None:
        assert _extract_citations([]) == []

    def test_extracts_s3_uri(self) -> None:
        raw = [
            {
                "retrievedReferences": [
                    {
                        "content": {"text": "some chunk"},
                        "location": {
                            "type": "S3",
                            "s3Location": {"uri": "s3://bucket/docs/paper.pdf"},
                        },
                        "metadata": {"title": "My Paper"},
                    }
                ]
            }
        ]
        result = _extract_citations(raw)
        assert len(result) == 1
        assert result[0]["url"] == "s3://bucket/docs/paper.pdf"
        assert result[0]["title"] == "My Paper"

    def test_deduplicates_by_uri(self) -> None:
        ref = {
            "content": {"text": "chunk"},
            "location": {"s3Location": {"uri": "s3://bucket/doc.pdf"}},
            "metadata": {},
        }
        raw = [
            {"retrievedReferences": [ref]},
            {"retrievedReferences": [ref]},
        ]
        result = _extract_citations(raw)
        assert len(result) == 1

    def test_falls_back_to_filename_title(self) -> None:
        raw = [
            {
                "retrievedReferences": [
                    {
                        "content": {"text": "chunk"},
                        "location": {"s3Location": {"uri": "s3://b/path/report.pdf"}},
                        "metadata": {},
                    }
                ]
            }
        ]
        result = _extract_citations(raw)
        assert result[0]["title"] == "report.pdf"

    def test_web_location(self) -> None:
        raw = [
            {
                "retrievedReferences": [
                    {
                        "content": {"text": "chunk"},
                        "location": {
                            "type": "WEB",
                            "webLocation": {"url": "https://example.com/page"},
                        },
                        "metadata": {"title": "Web Page"},
                    }
                ]
            }
        ]
        result = _extract_citations(raw)
        assert result[0]["url"] == "https://example.com/page"


# ---------------------------------------------------------------------------
# BedrockKBClient
# ---------------------------------------------------------------------------


class TestBedrockKBClient:
    def test_raises_without_kb_id(self) -> None:
        cfg = LibrarySettings(bedrock_knowledge_base_id="")
        with pytest.raises(ValueError, match="not configured"):
            BedrockKBClient(cfg)

    @patch("librarian.bedrock.client.boto3")
    def test_query_returns_response(self, mock_boto3: MagicMock) -> None:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.retrieve_and_generate.return_value = {
            "output": {"text": "The answer."},
            "citations": [
                {
                    "retrievedReferences": [
                        {
                            "content": {"text": "chunk"},
                            "location": {"s3Location": {"uri": "s3://b/doc.pdf"}},
                            "metadata": {"title": "Doc"},
                        }
                    ]
                }
            ],
            "sessionId": "ses-123",
        }

        cfg = LibrarySettings(
            bedrock_knowledge_base_id="KB123",
            bedrock_model_arn="arn:aws:bedrock:us-east-1::foundation-model/test",
        )
        client = BedrockKBClient(cfg)
        result = client.query("test question")

        assert isinstance(result, BedrockKBResponse)
        assert result.response == "The answer."
        assert len(result.citations) == 1
        assert result.session_id == "ses-123"

        # Verify boto3 was called correctly
        mock_client.retrieve_and_generate.assert_called_once()
        call_kwargs = mock_client.retrieve_and_generate.call_args[1]
        assert call_kwargs["input"]["text"] == "test question"
        kb_config = call_kwargs["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]
        assert kb_config["knowledgeBaseId"] == "KB123"

    @patch("librarian.bedrock.client.boto3")
    def test_query_passes_session_id(self, mock_boto3: MagicMock) -> None:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.retrieve_and_generate.return_value = {
            "output": {"text": "Follow-up answer."},
            "citations": [],
            "sessionId": "ses-123",
        }

        cfg = LibrarySettings(
            bedrock_knowledge_base_id="KB123",
            bedrock_model_arn="arn:aws:bedrock:us-east-1::foundation-model/test",
        )
        client = BedrockKBClient(cfg)
        client.query("follow up", session_id="ses-123")

        call_kwargs = mock_client.retrieve_and_generate.call_args[1]
        assert call_kwargs["sessionId"] == "ses-123"


# ---------------------------------------------------------------------------
# API integration — /chat with backend=bedrock
# ---------------------------------------------------------------------------


def _make_mock_bedrock_client() -> MagicMock:
    mock = MagicMock(spec=BedrockKBClient)
    mock.aquery = AsyncMock(
        return_value=BedrockKBResponse(
            response="Bedrock says hello.",
            citations=[{"url": "s3://b/doc.pdf", "title": "Doc"}],
            session_id="ses-456",
        )
    )
    return mock


@pytest.fixture()
def _mock_graph() -> Any:
    mock = AsyncMock()
    mock.ainvoke = AsyncMock(
        return_value={
            "response": "Librarian answer.",
            "citations": [],
            "confidence_score": 0.9,
            "intent": "lookup",
        }
    )
    mock.astream = AsyncMock()
    return mock


@pytest.fixture()
def client(_mock_graph: Any) -> TestClient:
    """TestClient with both graph and bedrock mocks."""
    deps._graph = _mock_graph
    deps._generation_sg = AsyncMock()
    deps._pipeline = AsyncMock()
    deps._bedrock_client = _make_mock_bedrock_client()
    yield TestClient(app, raise_server_exceptions=True)
    deps._graph = None
    deps._generation_sg = None
    deps._pipeline = None
    deps._bedrock_client = None


class TestChatBackendRouting:
    def test_default_backend_is_librarian(self, client: TestClient) -> None:
        resp = client.post("/api/v1/chat", json={"query": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "librarian"
        assert data["response"] == "Librarian answer."

    def test_explicit_librarian_backend(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/chat", json={"query": "hello", "backend": "librarian"}
        )
        assert resp.status_code == 200
        assert resp.json()["backend"] == "librarian"

    def test_bedrock_backend(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/chat", json={"query": "hello", "backend": "bedrock"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "bedrock"
        assert data["response"] == "Bedrock says hello."
        assert len(data["citations"]) == 1

    def test_bedrock_not_configured(self, client: TestClient) -> None:
        deps._bedrock_client = None
        resp = client.post(
            "/api/v1/chat", json={"query": "hello", "backend": "bedrock"}
        )
        assert resp.status_code == 503
        assert "not configured" in resp.json()["error"]
