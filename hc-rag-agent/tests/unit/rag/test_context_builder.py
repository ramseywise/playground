"""Context builder — token budget and deduplication."""

from __future__ import annotations

import pytest

from orchestrator.langgraph.schemas.context_builder import (
    ContextBuildConfig,
    build_context_from_ranked,
)
from rag.schemas.chunks import Chunk, ChunkMetadata, RankedChunk


@pytest.fixture(autouse=True)
def _approx_token_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid tiktoken blob download in CI/sandbox — rough length-based stand-in."""

    def _fake(text: str) -> int:
        return 0 if not text else max(1, len(text) // 4)

    monkeypatch.setattr(
        "src.orchestrator.langgraph.schemas.context_builder.count_tokens", _fake
    )


def _chunk(cid: str, text: str, rank: int, score: float) -> RankedChunk:
    return RankedChunk(
        chunk=Chunk(id=cid, text=text, metadata=ChunkMetadata(title=cid)),
        relevance_score=score,
        rank=rank,
    )


def test_empty_ranked() -> None:
    r = build_context_from_ranked([])
    assert "No relevant documents" in r.text
    assert r.chunk_ids_in_order == ()
    assert not r.truncated


def test_dedupe_by_chunk_id() -> None:
    a = _chunk("1", "hello world", 1, 0.9)
    b = _chunk("1", "duplicate id", 2, 0.8)
    r = build_context_from_ranked(
        [a, b], ContextBuildConfig(max_tokens=100_000, max_chunks=10)
    )
    assert r.chunk_ids_in_order == ("1",)


def test_respects_max_chunks() -> None:
    chunks = [_chunk(str(i), "x" * 50, i, 0.5) for i in range(1, 8)]
    r = build_context_from_ranked(
        chunks, ContextBuildConfig(max_tokens=100_000, max_chunks=3)
    )
    assert len(r.chunk_ids_in_order) == 3
    assert r.truncated
