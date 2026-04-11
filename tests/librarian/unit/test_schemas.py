from __future__ import annotations

import pytest
from pydantic import ValidationError

from librarian.schemas.chunks import (
    Chunk,
    ChunkMetadata,
    GradedChunk,
    RankedChunk,
)
from librarian.schemas.retrieval import Intent, QueryPlan, RetrievalResult
from librarian.schemas.state import LibrarianState


def _make_metadata(**kwargs: object) -> ChunkMetadata:
    return ChunkMetadata(
        url="https://example.com/doc", title="Doc", doc_id="doc-1", **kwargs
    )


def _make_chunk(**kwargs: object) -> Chunk:
    return Chunk(id="chunk-1", text="hello world", metadata=_make_metadata(), **kwargs)


# ---------------------------------------------------------------------------
# ChunkMetadata
# ---------------------------------------------------------------------------


def test_chunk_metadata_defaults() -> None:
    m = _make_metadata()
    assert m.language == "en"
    assert m.namespace is None
    assert m.completeness_score is None


def test_chunk_metadata_optional_fields() -> None:
    m = _make_metadata(
        namespace="docs", topic="auth", access_tier="public", completeness_score=0.95
    )
    assert m.namespace == "docs"
    assert m.completeness_score == 0.95


def test_chunk_metadata_requires_url_title_doc_id() -> None:
    with pytest.raises(ValidationError):
        ChunkMetadata(url="https://example.com")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


def test_chunk_embedding_optional() -> None:
    c = _make_chunk()
    assert c.embedding is None


def test_chunk_with_embedding() -> None:
    c = _make_chunk(embedding=[0.1, 0.2, 0.3])
    assert len(c.embedding) == 3  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# GradedChunk
# ---------------------------------------------------------------------------


def test_graded_chunk_fields() -> None:
    gc = GradedChunk(chunk=_make_chunk(), score=0.82, relevant=True)
    assert gc.relevant is True
    assert gc.score == 0.82


# ---------------------------------------------------------------------------
# RankedChunk
# ---------------------------------------------------------------------------


def test_ranked_chunk_score_bounds() -> None:
    rc = RankedChunk(chunk=_make_chunk(), relevance_score=0.75, rank=1)
    assert rc.rank == 1


def test_ranked_chunk_rejects_out_of_bounds() -> None:
    with pytest.raises(ValidationError):
        RankedChunk(chunk=_make_chunk(), relevance_score=1.5, rank=1)


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


def test_retrieval_result_source_literal() -> None:
    rr = RetrievalResult(chunk=_make_chunk(), score=0.9, source="hybrid")
    assert rr.source == "hybrid"


def test_retrieval_result_rejects_bad_source() -> None:
    with pytest.raises(ValidationError):
        RetrievalResult(chunk=_make_chunk(), score=0.9, source="unknown")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# QueryPlan
# ---------------------------------------------------------------------------


def test_query_plan_minimal() -> None:
    qp = QueryPlan(
        intent=Intent.LOOKUP,
        routing="retrieve",
        query_variants=["what is X?"],
        needs_clarification=False,
    )
    assert qp.clarification_question is None


def test_query_plan_with_clarification() -> None:
    qp = QueryPlan(
        intent=Intent.EXPLORE,
        routing="clarify",
        query_variants=[],
        needs_clarification=True,
        clarification_question="Which version are you asking about?",
    )
    assert qp.needs_clarification is True


def test_query_plan_rejects_bad_routing() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            intent=Intent.LOOKUP,
            routing="unknown",  # type: ignore[arg-type]
            query_variants=[],
            needs_clarification=False,
        )


# ---------------------------------------------------------------------------
# Intent enum
# ---------------------------------------------------------------------------


def test_intent_values() -> None:
    assert Intent.LOOKUP == "lookup"
    assert Intent.OUT_OF_SCOPE == "out_of_scope"


# ---------------------------------------------------------------------------
# LibrarianState — TypedDict field access + add_messages reducer
# ---------------------------------------------------------------------------


def test_librarian_state_partial_construction() -> None:
    state: LibrarianState = {"query": "What is X?", "retry_count": 0}
    assert state["query"] == "What is X?"
    assert "response" not in state


def test_librarian_state_messages_key_accepts_list() -> None:
    state: LibrarianState = {"messages": [{"role": "user", "content": "hi"}]}
    assert len(state["messages"]) == 1
    assert state["messages"][-1]["content"] == "hi"
