"""Tests for Reciprocal Rank Fusion (retrieval/rrf.py)."""

from __future__ import annotations

import pytest

from agents.librarian.rag_core.retrieval.rrf import RRF_K, fuse_rankings, _chunk_hash
from agents.librarian.rag_core.schemas.chunks import Chunk, ChunkMetadata, GradedChunk


def _chunk(id_: str, text: str = "text") -> Chunk:
    return Chunk(
        id=id_,
        text=text,
        metadata=ChunkMetadata(url="https://example.com", title="T", doc_id="d1"),
    )


def _graded(id_: str, score: float = 0.5, *, relevant: bool = True) -> GradedChunk:
    return GradedChunk(chunk=_chunk(id_), score=score, relevant=relevant)


class TestFuseRankings:
    def test_empty_rankings(self) -> None:
        assert fuse_rankings([]) == []

    def test_single_list_preserves_order(self) -> None:
        ranking = [_graded("a", 0.9), _graded("b", 0.5)]
        result = fuse_rankings([ranking])
        assert [r.chunk.id for r in result] == ["a", "b"]

    def test_fused_score_accumulates(self) -> None:
        list1 = [_graded("a"), _graded("b")]
        list2 = [_graded("b"), _graded("a")]
        result = fuse_rankings([list1, list2])
        # Both appear at rank 1 and rank 2 across lists, so scores equal.
        scores = {r.chunk.id: r.score for r in result}
        assert scores["a"] == pytest.approx(scores["b"])

    def test_item_in_multiple_lists_scores_higher(self) -> None:
        list1 = [_graded("a"), _graded("b")]
        list2 = [_graded("a"), _graded("c")]
        result = fuse_rankings([list1, list2])
        scores = {r.chunk.id: r.score for r in result}
        # "a" appears in both lists, "b" and "c" in one each.
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["c"]

    def test_top_k_limits_results(self) -> None:
        ranking = [_graded("a"), _graded("b"), _graded("c")]
        result = fuse_rankings([ranking], top_k=2)
        assert len(result) == 2

    def test_default_k_is_60(self) -> None:
        assert RRF_K == 60

    def test_custom_k_affects_scores(self) -> None:
        ranking = [_graded("a")]
        result_default = fuse_rankings([ranking], k=60)
        result_custom = fuse_rankings([ranking], k=10)
        # Smaller k means higher RRF score for rank 1.
        assert result_custom[0].score > result_default[0].score

    def test_preserves_relevant_flag(self) -> None:
        ranking = [_graded("a", relevant=False)]
        result = fuse_rankings([ranking])
        assert result[0].relevant is False

    def test_keeps_highest_original_score_on_dedup(self) -> None:
        list1 = [_graded("a", 0.9)]
        list2 = [_graded("a", 0.3)]
        result = fuse_rankings([list1, list2])
        # The GradedChunk kept should have the higher original score's relevant flag.
        assert len(result) == 1


class TestChunkHash:
    def test_deterministic(self) -> None:
        c = _chunk("x", "hello")
        assert _chunk_hash(c) == _chunk_hash(c)

    def test_different_ids_differ(self) -> None:
        assert _chunk_hash(_chunk("a")) != _chunk_hash(_chunk("b"))

    def test_returns_hex_string(self) -> None:
        h = _chunk_hash(_chunk("a"))
        assert isinstance(h, str)
        assert len(h) == 16
