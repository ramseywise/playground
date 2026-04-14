"""Tests for the ADK BedrockKBAgent wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.sessions import Session
from google.genai import types

from clients.bedrock import BedrockKBResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(query: str, session_id: str = "sess-123") -> Session:
    """Build a minimal ADK Session with one user event."""
    user_event = Event(
        author="user",
        content=types.Content(parts=[types.Part(text=query)]),
    )
    return Session(
        id=session_id,
        app_name="test",
        user_id="u1",
        events=[user_event],
    )


def _make_ctx(query: str, session_id: str = "sess-123") -> InvocationContext:
    """Build a minimal InvocationContext for testing."""
    session = _make_session(query, session_id)
    ctx = MagicMock(spec=InvocationContext)
    ctx.session = session
    return ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_bedrock_response() -> BedrockKBResponse:
    return BedrockKBResponse(
        response="Bedrock says: authentication uses OAuth2.",
        citations=[
            {"url": "https://docs.example.com/auth", "title": "Auth Docs"},
        ],
        session_id="sess-123",
    )


@pytest.mark.asyncio
async def test_bedrock_agent_emits_event(
    mock_bedrock_response: BedrockKBResponse,
) -> None:
    """Agent should yield an Event with the Bedrock response text."""
    with patch("orchestration.google_adk.bedrock_agent.BedrockKBClient") as MockClient:
        mock_client = MockClient.return_value
        mock_client.aquery = AsyncMock(return_value=mock_bedrock_response)

        from orchestration.google_adk.bedrock_agent import BedrockKBAgent
        from librarian.config import LibrarySettings

        cfg = LibrarySettings(
            bedrock_knowledge_base_id="kb-test-123",
            bedrock_model_arn="arn:aws:bedrock:us-east-1::foundation-model/test",
            anthropic_api_key="test",
        )
        agent = BedrockKBAgent(cfg)

        ctx = _make_ctx("what is authentication?")
        events = []
        async for event in agent._run_async_impl(ctx):
            events.append(event)

        assert len(events) == 1
        assert events[0].author == "bedrock_kb"
        assert events[0].content is not None
        assert mock_bedrock_response.response in events[0].content.parts[0].text
        # Citations should be appended as sources
        assert "Auth Docs" in events[0].content.parts[0].text
        assert events[0].custom_metadata["citations"] == mock_bedrock_response.citations


@pytest.mark.asyncio
async def test_bedrock_agent_passes_session_id(
    mock_bedrock_response: BedrockKBResponse,
) -> None:
    """Agent should pass session ID from ADK context to Bedrock."""
    with patch("orchestration.google_adk.bedrock_agent.BedrockKBClient") as MockClient:
        mock_client = MockClient.return_value
        mock_client.aquery = AsyncMock(return_value=mock_bedrock_response)

        from orchestration.google_adk.bedrock_agent import BedrockKBAgent
        from librarian.config import LibrarySettings

        cfg = LibrarySettings(
            bedrock_knowledge_base_id="kb-test-123",
            bedrock_model_arn="arn:aws:bedrock:us-east-1::foundation-model/test",
            anthropic_api_key="test",
        )
        agent = BedrockKBAgent(cfg)

        ctx = _make_ctx("what is auth?", session_id="my-session-42")
        async for _ in agent._run_async_impl(ctx):
            pass

        mock_client.aquery.assert_awaited_once_with(
            "what is auth?",
            session_id="my-session-42",
        )


@pytest.mark.asyncio
async def test_bedrock_agent_extracts_latest_user_message() -> None:
    """Agent should use the last user message, not the first."""
    resp = BedrockKBResponse(response="answer", citations=[], session_id="s1")

    with patch("orchestration.google_adk.bedrock_agent.BedrockKBClient") as MockClient:
        mock_client = MockClient.return_value
        mock_client.aquery = AsyncMock(return_value=resp)

        from orchestration.google_adk.bedrock_agent import BedrockKBAgent
        from librarian.config import LibrarySettings

        cfg = LibrarySettings(
            bedrock_knowledge_base_id="kb-test",
            bedrock_model_arn="arn:aws:bedrock:us-east-1::foundation-model/test",
            anthropic_api_key="test",
        )
        agent = BedrockKBAgent(cfg)

        # Session with multiple user messages
        events_list = [
            Event(
                author="user",
                content=types.Content(parts=[types.Part(text="first question")]),
            ),
            Event(
                author="bedrock_kb",
                content=types.Content(parts=[types.Part(text="first answer")]),
            ),
            Event(
                author="user",
                content=types.Content(parts=[types.Part(text="follow-up question")]),
            ),
        ]
        session = Session(
            id="sess-multi",
            app_name="test",
            user_id="u1",
            events=events_list,
        )
        ctx = MagicMock(spec=InvocationContext)
        ctx.session = session

        async for _ in agent._run_async_impl(ctx):
            pass

        # Should extract "follow-up question" (last user message)
        mock_client.aquery.assert_awaited_once_with(
            "follow-up question",
            session_id="sess-multi",
        )


@pytest.mark.asyncio
async def test_bedrock_agent_name_and_description() -> None:
    """Agent should have correct name and description."""
    with patch("orchestration.google_adk.bedrock_agent.BedrockKBClient"):
        from orchestration.google_adk.bedrock_agent import BedrockKBAgent
        from librarian.config import LibrarySettings

        cfg = LibrarySettings(
            bedrock_knowledge_base_id="kb-test",
            bedrock_model_arn="arn:aws:bedrock:us-east-1::foundation-model/test",
            anthropic_api_key="test",
        )
        agent = BedrockKBAgent(cfg)
        assert agent.name == "bedrock_kb"
        assert "Bedrock" in agent.description
