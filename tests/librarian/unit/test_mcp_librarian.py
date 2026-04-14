"""Tests for the Librarian MCP server."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from interfaces.mcp import librarian_server


@pytest.fixture(autouse=True)
def _reset_singletons() -> Any:
    """Reset module-level singletons between tests."""
    librarian_server._graph = None
    librarian_server._pipeline = None
    yield
    librarian_server._graph = None
    librarian_server._pipeline = None


class TestGetGraph:
    def test_lazy_init(self) -> None:
        mock_graph = MagicMock()
        cfg = MagicMock()

        with patch(
            "orchestration.factory.create_librarian",
            return_value=mock_graph,
        ) as mock_create:
            result = librarian_server._get_graph(cfg)

        assert result is mock_graph
        mock_create.assert_called_once_with(cfg)

    def test_returns_cached(self) -> None:
        mock_graph = MagicMock()
        librarian_server._graph = mock_graph
        cfg = MagicMock()

        result = librarian_server._get_graph(cfg)
        assert result is mock_graph


class TestGetPipeline:
    def test_lazy_init(self) -> None:
        mock_pipeline = MagicMock()
        cfg = MagicMock()

        with patch(
            "orchestration.factory.create_ingestion_pipeline",
            return_value=mock_pipeline,
        ) as mock_create:
            result = librarian_server._get_pipeline(cfg)

        assert result is mock_pipeline
        mock_create.assert_called_once_with(cfg)


class TestCreateServer:
    def test_server_name(self) -> None:
        cfg = MagicMock()
        server = librarian_server.create_server(cfg)
        assert server.name == "mcp-librarian"
