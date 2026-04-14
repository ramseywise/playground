"""Tests for the ADK LibrarianADKAgent (hybrid LangGraph + ADK wrapper)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.sessions import Session
from google.genai import types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(query: str, session_id: str = "sess-123") -> Session:
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
    session = _make_session(query, session_id)
    ctx = MagicMock(spec=InvocationContext)
    ctx.session = session
    return ctx


# ---------------------------------------------------------------------------
# LibrarianADKAgent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_agent_calls_graph_ainvoke() -> None:
    """Agent should call graph.ainvoke with the query and thread_id."""
    from orchestration.google_adk.hybrid_agent import LibrarianADKAgent

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "response": "OAuth2 is the standard.",
            "citations": [{"url": "https://docs.example.com"}],
            "confidence_score": 0.85,
        }
    )

    agent = LibrarianADKAgent(graph=mock_graph)
    ctx = _make_ctx("what is authentication?")

    events = []
    async for event in agent._run_async_impl(ctx):
        events.append(event)

    assert len(events) == 1
    assert events[0].author == "librarian_hybrid"
    assert events[0].content.parts[0].text == "OAuth2 is the standard."

    # Verify graph was called with correct state
    mock_graph.ainvoke.assert_awaited_once()
    call_args = mock_graph.ainvoke.call_args
    state = call_args[0][0]
    assert state["query"] == "what is authentication?"
    assert call_args[1]["config"]["configurable"]["thread_id"] == "sess-123"


@pytest.mark.asyncio
async def test_hybrid_agent_passes_messages() -> None:
    """Agent should extract conversation history from ADK session events."""
    from orchestration.google_adk.hybrid_agent import LibrarianADKAgent

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"response": "follow-up answer"})

    agent = LibrarianADKAgent(graph=mock_graph)

    # Multi-turn session
    events_list = [
        Event(
            author="user",
            content=types.Content(parts=[types.Part(text="first question")]),
        ),
        Event(
            author="librarian_hybrid",
            content=types.Content(parts=[types.Part(text="first answer")]),
        ),
        Event(
            author="user",
            content=types.Content(parts=[types.Part(text="follow-up")]),
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

    call_state = mock_graph.ainvoke.call_args[0][0]
    assert call_state["query"] == "follow-up"
    assert len(call_state["messages"]) == 3
    assert call_state["messages"][0]["role"] == "user"
    assert call_state["messages"][1]["role"] == "assistant"
    assert call_state["messages"][2]["role"] == "user"


@pytest.mark.asyncio
async def test_hybrid_agent_handles_empty_response() -> None:
    """Agent should handle empty graph response gracefully."""
    from orchestration.google_adk.hybrid_agent import LibrarianADKAgent

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"response": ""})

    agent = LibrarianADKAgent(graph=mock_graph)
    ctx = _make_ctx("test query")

    events = []
    async for event in agent._run_async_impl(ctx):
        events.append(event)

    assert len(events) == 1
    assert events[0].content.parts[0].text == ""


def test_hybrid_agent_name_and_description() -> None:
    """Agent should have correct name and description."""
    from orchestration.google_adk.hybrid_agent import LibrarianADKAgent

    agent = LibrarianADKAgent(graph=MagicMock())
    assert agent.name == "librarian_hybrid"
    assert "CRAG" in agent.description


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


def test_coordinator_has_both_sub_agents() -> None:
    """Coordinator should have both sub-agents wired."""
    from google.adk.agents import Agent

    from orchestration.google_adk.coordinator import create_coordinator

    mock_librarian = Agent(
        model="gemini-2.0-flash",
        name="librarian_hybrid",
        description="test librarian",
        instruction="test",
    )
    mock_custom_rag = Agent(
        model="gemini-2.0-flash",
        name="custom_rag",
        description="test custom rag",
        instruction="test",
    )

    coordinator = create_coordinator(mock_librarian, mock_custom_rag)

    assert coordinator.name == "coordinator"
    sub_agent_names = [a.name for a in coordinator.sub_agents]
    assert "librarian_hybrid" in sub_agent_names
    assert "custom_rag" in sub_agent_names
