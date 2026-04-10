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
    """Mock LLM with ``generate(system, messages) -> str`` interface.

    Usage in tests:
        mock_llm.generate.return_value = "the answer"
    """
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm
