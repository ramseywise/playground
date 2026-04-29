"""Unit tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from rag.retrieval.rrf import chunk_fingerprint as _chunk_hash, fuse_rankings
from rag.schemas.chunks import Chunk, ChunkMetadata, GradedChunk


def _make_gc(url: str, text: str, score: float = 1.0) -> GradedChunk:
    meta = ChunkMetadata(url=url)
    chunk = Chunk(id=f"{url}-id", text=text, metadata=meta)
    return GradedChunk(chunk=chunk, score=score)


class TestChunkHash:
    def test_same_chunk_same_hash(self) -> None:
        gc = _make_gc("https://example.com", "hello world")
        assert _chunk_hash(gc) == _chunk_hash(gc)

    def test_different_url_different_hash(self) -> None:
        gc1 = _make_gc("https://a.com", "hello")
        gc2 = _make_gc("https://b.com", "hello")
        assert _chunk_hash(gc1) != _chunk_hash(gc2)

    def test_different_text_different_hash(self) -> None:
        gc1 = _make_gc("https://x.com", "foo")
        gc2 = _make_gc("https://x.com", "bar")
        assert _chunk_hash(gc1) != _chunk_hash(gc2)

    def test_hash_is_16_hex_chars(self) -> None:
        gc = _make_gc("https://example.com", "some text")
        h = _chunk_hash(gc)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestFuseRankings:
    def test_empty_input_returns_empty(self) -> None:
        assert fuse_rankings([]) == []

    def test_single_list_preserves_order(self) -> None:
        a = _make_gc("https://a.com", "alpha", score=0.9)
        b = _make_gc("https://b.com", "beta", score=0.8)
        c = _make_gc("https://c.com", "gamma", score=0.7)
        result = fuse_rankings([[a, b, c]])
        assert [r.chunk.metadata.url for r in result] == [
            "https://a.com",
            "https://b.com",
            "https://c.com",
        ]

    def test_chunk_appearing_in_two_lists_ranks_higher(self) -> None:
        shared = _make_gc("https://shared.com", "shared content")
        only_list1 = _make_gc("https://only1.com", "unique to list 1")
        only_list2 = _make_gc("https://only2.com", "unique to list 2")

        # shared appears rank-1 in both lists → boosted score
        result = fuse_rankings([[shared, only_list1], [shared, only_list2]])
        urls = [r.chunk.metadata.url for r in result]
        assert urls[0] == "https://shared.com"

    def test_deduplication_keeps_higher_score(self) -> None:
        low_score = _make_gc("https://dup.com", "duplicate text", score=0.3)
        high_score = _make_gc("https://dup.com", "duplicate text", score=0.9)

        result = fuse_rankings([[low_score], [high_score]])
        # only one entry for the deduplicated chunk
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9)

    def test_k_parameter_affects_scores_not_ordering(self) -> None:
        a = _make_gc("https://a.com", "first")
        b = _make_gc("https://b.com", "second")
        # With a single list, order shouldn't flip regardless of k
        result_small_k = fuse_rankings([[a, b]], k=1)
        result_large_k = fuse_rankings([[a, b]], k=1000)
        assert (
            result_small_k[0].chunk.metadata.url == result_large_k[0].chunk.metadata.url
        )

    def test_returns_all_unique_chunks(self) -> None:
        chunks = [_make_gc(f"https://c{i}.com", f"text {i}") for i in range(5)]
        result = fuse_rankings([chunks[:3], chunks[2:]])
        assert len(result) == 5
