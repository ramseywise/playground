from __future__ import annotations

from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.librarian.retrieval.inmemory import InMemoryRetriever
from agents.librarian.retrieval.mock_embedder import MockEmbedder

# ---------------------------------------------------------------------------
# Registry isolation — runs before/after every test automatically
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_registry() -> Generator[None, None, None]:
    from agents.librarian.utils.registry import Registry

    Registry.clear()
    yield
    Registry.clear()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_embedder() -> MockEmbedder:
    return MockEmbedder(dim=1024, seed=42)


@pytest.fixture()
def inmemory_retriever() -> InMemoryRetriever:
    return InMemoryRetriever()


@pytest.fixture()
def mock_llm() -> MagicMock:
    """Patches ChatAnthropic.ainvoke with a configurable AsyncMock.

    Usage in tests:
        mock_llm.ainvoke.return_value = AIMessage(content="...")
    """
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    llm.with_structured_output = MagicMock(return_value=llm)
    return llm
