from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.librarian.tools.storage.vectordb.inmemory import InMemoryRetriever
from tests.librarian.testing.mock_embedder import MockEmbedder

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
    """Mock LLM with ``generate(system, messages) -> str`` interface.

    Usage in tests:
        mock_llm.generate.return_value = "the answer"
    """
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm
