from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestration.history import HistoryCondenser
from librarian.schemas.state import LibrarianState


def _mock_llm(response: str = "standalone query") -> MagicMock:
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=response)
    return llm


@pytest.mark.asyncio
async def test_history_condenser_passes_through_single_turn() -> None:
    condenser = HistoryCondenser(llm=_mock_llm())
    state: LibrarianState = {
        "query": "what is auth?",
        "messages": [{"role": "user", "content": "what is auth?"}],
    }
    result = await condenser.condense(state)
    assert result["standalone_query"] == "what is auth?"


@pytest.mark.asyncio
async def test_history_condenser_rewrites_multi_turn() -> None:
    llm = _mock_llm("what is the auth flow in Python?")
    condenser = HistoryCondenser(llm=llm)
    state: LibrarianState = {
        "query": "and for Python?",
        "messages": [
            {"role": "user", "content": "what is the auth flow?"},
            {"role": "assistant", "content": "It uses OAuth."},
            {"role": "user", "content": "and for Python?"},
        ],
    }
    result = await condenser.condense(state)
    assert result["standalone_query"] == "what is the auth flow in Python?"
    llm.generate.assert_awaited_once()
