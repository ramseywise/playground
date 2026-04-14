"""Tests for the tool abstraction layer (librarian/tools/)."""

from __future__ import annotations

import pytest

from librarian.retrieval.ensemble import EnsembleRetriever
from librarian.schemas.chunks import Chunk, ChunkMetadata
from librarian.tools.base import BaseTool, ToolInput, ToolOutput
from librarian.tools.retriever_tool import (
    RetrieverTool,
    RetrieverToolInput,
    RetrieverToolOutput,
)
from storage.vectordb.inmemory import InMemoryRetriever
from tests.librarian.testing.mock_embedder import MockEmbedder


def _seeded_chunks(n: int = 5) -> list[Chunk]:
    embedder = MockEmbedder(dim=8, seed=99)
    chunks = []
    for i in range(n):
        text = f"chunk number {i} with unique content"
        c = Chunk(
            id=f"c{i}",
            text=text,
            metadata=ChunkMetadata(
                url=f"https://example.com/doc{i}", title=f"Doc {i}", doc_id="d1"
            ),
        )
        c.embedding = embedder.embed_passage(text)
        chunks.append(c)
    return chunks


# ---------------------------------------------------------------------------
# BaseTool protocol
# ---------------------------------------------------------------------------


class TestBaseToolProtocol:
    def test_retriever_tool_satisfies_protocol(self) -> None:
        retriever = InMemoryRetriever()
        embedder = MockEmbedder(dim=8, seed=42)
        ensemble = EnsembleRetriever([retriever], embedder)
        tool = RetrieverTool(ensemble)
        assert isinstance(tool, BaseTool)

    def test_tool_has_required_attributes(self) -> None:
        retriever = InMemoryRetriever()
        embedder = MockEmbedder(dim=8, seed=42)
        ensemble = EnsembleRetriever([retriever], embedder)
        tool = RetrieverTool(ensemble)
        assert tool.name == "search_knowledge_base"
        assert len(tool.description) > 0
        assert issubclass(tool.input_schema, ToolInput)
        assert issubclass(tool.output_schema, ToolOutput)


# ---------------------------------------------------------------------------
# RetrieverToolInput validation
# ---------------------------------------------------------------------------


class TestRetrieverToolInput:
    def test_valid_input(self) -> None:
        inp = RetrieverToolInput(queries=["hello"], num_results=5)
        assert inp.queries == ["hello"]
        assert inp.num_results == 5

    def test_default_num_results(self) -> None:
        inp = RetrieverToolInput(queries=["q"])
        assert inp.num_results == 10

    def test_rejects_empty_queries(self) -> None:
        with pytest.raises(Exception):
            RetrieverToolInput(queries=[])

    def test_rejects_too_many_queries(self) -> None:
        with pytest.raises(Exception):
            RetrieverToolInput(queries=["a", "b", "c", "d"])

    def test_rejects_zero_results(self) -> None:
        with pytest.raises(Exception):
            RetrieverToolInput(queries=["q"], num_results=0)


# ---------------------------------------------------------------------------
# RetrieverTool.run()
# ---------------------------------------------------------------------------


class TestRetrieverTool:
    @pytest.mark.asyncio()
    async def test_returns_output_model(self) -> None:
        retriever = InMemoryRetriever()
        embedder = MockEmbedder(dim=8, seed=42)
        chunks = _seeded_chunks(3)
        await retriever.upsert(chunks)

        ensemble = EnsembleRetriever(
            [retriever], embedder, score_threshold=0.0
        )
        tool = RetrieverTool(ensemble)

        result = await tool.run(RetrieverToolInput(queries=["chunk number 0"]))
        assert isinstance(result, RetrieverToolOutput)
        assert result.total > 0
        assert result.total == len(result.results)

    @pytest.mark.asyncio()
    async def test_result_structure(self) -> None:
        retriever = InMemoryRetriever()
        embedder = MockEmbedder(dim=8, seed=42)
        chunks = _seeded_chunks(3)
        await retriever.upsert(chunks)

        ensemble = EnsembleRetriever(
            [retriever], embedder, score_threshold=0.0
        )
        tool = RetrieverTool(ensemble)
        result = await tool.run(RetrieverToolInput(queries=["chunk"]))

        for r in result.results:
            assert "text" in r
            assert "url" in r
            assert "title" in r
            assert "score" in r
            assert "chunk_id" in r

    @pytest.mark.asyncio()
    async def test_empty_index_returns_zero(self) -> None:
        retriever = InMemoryRetriever()
        embedder = MockEmbedder(dim=8, seed=42)
        ensemble = EnsembleRetriever(
            [retriever], embedder, score_threshold=0.0
        )
        tool = RetrieverTool(ensemble)
        result = await tool.run(RetrieverToolInput(queries=["anything"]))
        assert result.total == 0

    @pytest.mark.asyncio()
    async def test_multi_query(self) -> None:
        retriever = InMemoryRetriever()
        embedder = MockEmbedder(dim=8, seed=42)
        chunks = _seeded_chunks(5)
        await retriever.upsert(chunks)

        ensemble = EnsembleRetriever(
            [retriever], embedder, score_threshold=0.0
        )
        tool = RetrieverTool(ensemble)
        result = await tool.run(
            RetrieverToolInput(queries=["chunk number 0", "chunk number 3"])
        )
        assert result.total > 0

    @pytest.mark.asyncio()
    async def test_serializable_output(self) -> None:
        retriever = InMemoryRetriever()
        embedder = MockEmbedder(dim=8, seed=42)
        chunks = _seeded_chunks(2)
        await retriever.upsert(chunks)

        ensemble = EnsembleRetriever(
            [retriever], embedder, score_threshold=0.0
        )
        tool = RetrieverTool(ensemble)
        result = await tool.run(RetrieverToolInput(queries=["chunk"]))
        # Output should be serializable (for ADK JSON responses)
        d = result.model_dump()
        assert isinstance(d, dict)
        assert "results" in d
