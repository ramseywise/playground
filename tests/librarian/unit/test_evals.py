"""Unit tests for eval_harness/tasks/extract_golden.py and generate_synthetic.py.

All LLM calls are mocked — no real API calls or file I/O beyond temp files.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from librarian.ingestion.tasks.extract_golden import (
    _make_query_id,
    extract_samples,
    filter_by_tier,
    load_records,
    load_samples,
    save_samples,
)
from librarian.ingestion.tasks.generate_synthetic import (
    CONFIRM_EXPENSIVE_OPS,
    generate_from_chunks,
)
from librarian.ingestion.tasks.models import GoldenSample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def silver_records() -> list[dict]:
    return [
        {
            "query": "how do I reset my password?",
            "expected_doc_url": "https://docs.example.com/password",
            "relevant_chunk_ids": ["c1"],
            "category": "auth",
            "language": "en",
            "difficulty": "easy",
            "validation_level": "silver",
        },
        {
            "query": "what is the rate limit?",
            "expected_doc_url": "https://docs.example.com/rate-limits",
            "relevant_chunk_ids": [],
            "category": "api",
            "language": "en",
            "difficulty": "medium",
            "validation_level": "silver",
        },
    ]


@pytest.fixture()
def mixed_tier_records() -> list[dict]:
    return [
        {
            "query": "gold query",
            "expected_doc_url": "https://docs.example.com/gold",
            "validation_level": "gold",
        },
        {
            "query": "silver query",
            "expected_doc_url": "https://docs.example.com/silver",
            "validation_level": "silver",
        },
        {
            "query": "bronze query",
            "expected_doc_url": "https://docs.example.com/bronze",
            "validation_level": "bronze",
        },
    ]


# ---------------------------------------------------------------------------
# _make_query_id
# ---------------------------------------------------------------------------


def test_make_query_id_is_deterministic() -> None:
    qid1 = _make_query_id("hello?", "https://example.com")
    qid2 = _make_query_id("hello?", "https://example.com")
    assert qid1 == qid2


def test_make_query_id_is_16_chars() -> None:
    qid = _make_query_id("query", "https://example.com")
    assert len(qid) == 16


def test_make_query_id_case_insensitive() -> None:
    qid1 = _make_query_id("HELLO?", "https://example.com/PATH")
    qid2 = _make_query_id("hello?", "https://example.com/path")
    assert qid1 == qid2


def test_make_query_id_differs_for_different_inputs() -> None:
    qid1 = _make_query_id("query A", "https://a.com")
    qid2 = _make_query_id("query B", "https://b.com")
    assert qid1 != qid2


# ---------------------------------------------------------------------------
# extract_samples — happy path
# ---------------------------------------------------------------------------


def test_extract_samples_returns_golden_samples(silver_records: list[dict]) -> None:
    samples = extract_samples(silver_records, tier="silver")
    assert len(samples) == 2
    assert all(isinstance(s, GoldenSample) for s in samples)


def test_extract_samples_sets_validation_level(silver_records: list[dict]) -> None:
    samples = extract_samples(silver_records, tier="silver")
    assert all(s.validation_level == "silver" for s in samples)


def test_extract_samples_copies_fields(silver_records: list[dict]) -> None:
    samples = extract_samples(silver_records, tier="silver")
    assert samples[0].query == "how do I reset my password?"
    assert samples[0].expected_doc_url == "https://docs.example.com/password"
    assert samples[0].category == "auth"
    assert samples[0].language == "en"
    assert samples[0].relevant_chunk_ids == ["c1"]


# ---------------------------------------------------------------------------
# extract_samples — filtering
# ---------------------------------------------------------------------------


def test_extract_samples_filters_by_tier(mixed_tier_records: list[dict]) -> None:
    gold = extract_samples(mixed_tier_records, tier="gold")
    silver = extract_samples(mixed_tier_records, tier="silver")
    bronze = extract_samples(mixed_tier_records, tier="bronze")
    assert len(gold) == 1
    assert len(silver) == 1
    assert len(bronze) == 1


def test_extract_samples_skips_missing_query() -> None:
    records = [
        {"expected_doc_url": "https://example.com", "validation_level": "silver"}
    ]
    samples = extract_samples(records, tier="silver")
    assert len(samples) == 0


def test_extract_samples_skips_missing_url() -> None:
    records = [{"query": "what?", "validation_level": "silver"}]
    samples = extract_samples(records, tier="silver")
    assert len(samples) == 0


def test_extract_samples_unknown_tier_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tier"):
        extract_samples([], tier="platinum")


# ---------------------------------------------------------------------------
# extract_samples — deduplication
# ---------------------------------------------------------------------------


def test_extract_samples_deduplicates() -> None:
    records = [
        {
            "query": "same query",
            "expected_doc_url": "https://a.com",
            "validation_level": "silver",
        },
        {
            "query": "same query",
            "expected_doc_url": "https://a.com",
            "validation_level": "silver",
        },
    ]
    samples = extract_samples(records, tier="silver")
    assert len(samples) == 1


def test_extract_samples_keeps_different_urls() -> None:
    records = [
        {
            "query": "same query",
            "expected_doc_url": "https://a.com",
            "validation_level": "silver",
        },
        {
            "query": "same query",
            "expected_doc_url": "https://b.com",
            "validation_level": "silver",
        },
    ]
    samples = extract_samples(records, tier="silver")
    assert len(samples) == 2


# ---------------------------------------------------------------------------
# extract_samples — records without validation_level default to tier
# ---------------------------------------------------------------------------


def test_extract_samples_accepts_records_without_validation_level() -> None:
    records = [{"query": "what?", "expected_doc_url": "https://example.com"}]
    samples = extract_samples(records, tier="silver")
    assert len(samples) == 1
    assert samples[0].validation_level == "silver"


# ---------------------------------------------------------------------------
# filter_by_tier
# ---------------------------------------------------------------------------


def test_filter_by_tier_single() -> None:
    samples = [
        GoldenSample(
            query_id="q1", query="q", expected_doc_url="u", validation_level="gold"
        ),
        GoldenSample(
            query_id="q2", query="q", expected_doc_url="u", validation_level="silver"
        ),
        GoldenSample(
            query_id="q3", query="q", expected_doc_url="u", validation_level="bronze"
        ),
    ]
    filtered = filter_by_tier(samples, ["gold"])
    assert len(filtered) == 1
    assert filtered[0].validation_level == "gold"


def test_filter_by_tier_multiple() -> None:
    samples = [
        GoldenSample(
            query_id="q1", query="q", expected_doc_url="u", validation_level="gold"
        ),
        GoldenSample(
            query_id="q2", query="q", expected_doc_url="u", validation_level="silver"
        ),
        GoldenSample(
            query_id="q3", query="q", expected_doc_url="u", validation_level="bronze"
        ),
    ]
    filtered = filter_by_tier(samples, ["gold", "silver"])
    assert len(filtered) == 2


def test_filter_by_tier_empty_result() -> None:
    samples = [
        GoldenSample(
            query_id="q1", query="q", expected_doc_url="u", validation_level="bronze"
        ),
    ]
    filtered = filter_by_tier(samples, ["gold"])
    assert filtered == []


# ---------------------------------------------------------------------------
# load_records — file parsing
# ---------------------------------------------------------------------------


def test_load_records_reads_jsonl(tmp_path: Path) -> None:
    records = [
        {"query": "q1", "expected_doc_url": "u1"},
        {"query": "q2", "expected_doc_url": "u2"},
    ]
    f = tmp_path / "records.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    loaded = load_records(f)
    assert len(loaded) == 2
    assert loaded[0]["query"] == "q1"


def test_load_records_skips_bad_json(tmp_path: Path) -> None:
    f = tmp_path / "bad.jsonl"
    f.write_text('{"query": "ok"}\nnot json\n{"query": "ok2"}\n')
    loaded = load_records(f)
    assert len(loaded) == 2


def test_load_records_skips_blank_lines(tmp_path: Path) -> None:
    f = tmp_path / "blanks.jsonl"
    f.write_text('{"query": "q1"}\n\n\n{"query": "q2"}\n')
    loaded = load_records(f)
    assert len(loaded) == 2


# ---------------------------------------------------------------------------
# save_samples / load_samples — round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path, silver_records: list[dict]) -> None:
    samples = extract_samples(silver_records, tier="silver")
    out = tmp_path / "golden.jsonl"
    save_samples(samples, out)
    loaded = load_samples(out)
    assert len(loaded) == len(samples)
    assert loaded[0].query == samples[0].query
    assert loaded[0].expected_doc_url == samples[0].expected_doc_url
    assert loaded[0].validation_level == "silver"


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    sample = GoldenSample(query_id="q1", query="q", expected_doc_url="u")
    nested = tmp_path / "deep" / "nested" / "golden.jsonl"
    save_samples([sample], nested)
    assert nested.exists()


def test_golden_sample_defaults_are_eval_friendly() -> None:
    sample = GoldenSample(query_id="q1", query="q", expected_doc_url="u")
    assert sample.language == "en"
    assert sample.difficulty == "medium"
    assert sample.validation_level == "silver"
    assert sample.source_record_id is None


# ---------------------------------------------------------------------------
# generate_from_chunks — cost gate
# ---------------------------------------------------------------------------


def test_generate_from_chunks_raises_without_cost_gate() -> None:
    assert CONFIRM_EXPENSIVE_OPS is False
    with pytest.raises(RuntimeError, match="CONFIRM_EXPENSIVE_OPS"):
        generate_from_chunks([{"text": "hello", "url": "https://example.com"}])


# ---------------------------------------------------------------------------
# generate_from_chunks — with cost gate patched
# ---------------------------------------------------------------------------


def _mock_anthropic_response(
    query: str = "what is auth?", difficulty: str = "easy"
) -> MagicMock:
    content_block = MagicMock()
    content_block.text = json.dumps({"query": query, "difficulty": difficulty})
    resp = MagicMock()
    resp.content = [content_block]
    return resp


def test_generate_from_chunks_returns_golden_samples() -> None:
    chunks = [
        {
            "text": "API keys are used for authentication",
            "url": "https://docs.example.com/auth",
            "chunk_id": "c1",
        },
    ]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response(
        "what are API keys used for?"
    )

    with (
        patch(
            "librarian.tasks.generate_synthetic.CONFIRM_EXPENSIVE_OPS",
            True,
        ),
        patch(
            "librarian.tasks.generate_synthetic.anthropic.Anthropic",
            return_value=mock_client,
        ),
    ):
        samples = generate_from_chunks(chunks)

    assert len(samples) == 1
    assert isinstance(samples[0], GoldenSample)
    assert samples[0].validation_level == "synthetic"
    assert samples[0].expected_doc_url == "https://docs.example.com/auth"
    assert samples[0].relevant_chunk_ids == ["c1"]


def test_generate_from_chunks_n_limits_output() -> None:
    chunks = [
        {"text": f"text {i}", "url": f"https://example.com/{i}", "chunk_id": f"c{i}"}
        for i in range(5)
    ]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response("q?")

    with (
        patch(
            "librarian.tasks.generate_synthetic.CONFIRM_EXPENSIVE_OPS",
            True,
        ),
        patch(
            "librarian.tasks.generate_synthetic.anthropic.Anthropic",
            return_value=mock_client,
        ),
    ):
        samples = generate_from_chunks(chunks, n=2)

    assert len(samples) == 2
    assert mock_client.messages.create.call_count == 2


def test_generate_from_chunks_skips_missing_text() -> None:
    chunks = [
        {"url": "https://example.com", "chunk_id": "c1"},  # no text
    ]
    mock_client = MagicMock()

    with (
        patch(
            "librarian.tasks.generate_synthetic.CONFIRM_EXPENSIVE_OPS",
            True,
        ),
        patch(
            "librarian.tasks.generate_synthetic.anthropic.Anthropic",
            return_value=mock_client,
        ),
    ):
        samples = generate_from_chunks(chunks)

    assert len(samples) == 0
    mock_client.messages.create.assert_not_called()


def test_generate_from_chunks_handles_parse_error() -> None:
    chunks = [{"text": "hello", "url": "https://example.com", "chunk_id": "c1"}]
    bad_response = MagicMock()
    bad_response.content = [MagicMock(text="not json")]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = bad_response

    with (
        patch(
            "librarian.tasks.generate_synthetic.CONFIRM_EXPENSIVE_OPS",
            True,
        ),
        patch(
            "librarian.tasks.generate_synthetic.anthropic.Anthropic",
            return_value=mock_client,
        ),
    ):
        samples = generate_from_chunks(chunks)

    assert samples == []


def test_generate_from_chunks_handles_api_error() -> None:
    import anthropic as anthropic_mod

    chunks = [{"text": "hello", "url": "https://example.com", "chunk_id": "c1"}]
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = anthropic_mod.APIError(
        message="rate limited", request=MagicMock(), body=None
    )

    with (
        patch(
            "librarian.tasks.generate_synthetic.CONFIRM_EXPENSIVE_OPS",
            True,
        ),
        patch(
            "librarian.tasks.generate_synthetic.anthropic.Anthropic",
            return_value=mock_client,
        ),
    ):
        samples = generate_from_chunks(chunks)

    assert samples == []


def test_generate_from_chunks_query_id_is_deterministic() -> None:
    chunks = [{"text": "hello", "url": "https://example.com", "chunk_id": "c1"}]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response(
        "what is hello?"
    )

    with (
        patch(
            "librarian.tasks.generate_synthetic.CONFIRM_EXPENSIVE_OPS",
            True,
        ),
        patch(
            "librarian.tasks.generate_synthetic.anthropic.Anthropic",
            return_value=mock_client,
        ),
    ):
        s1 = generate_from_chunks(chunks)
        s2 = generate_from_chunks(chunks)

    assert s1[0].query_id == s2[0].query_id


def test_generate_from_chunks_difficulty_propagated() -> None:
    chunks = [{"text": "hello", "url": "https://example.com", "chunk_id": "c1"}]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response(
        "q?", difficulty="hard"
    )

    with (
        patch(
            "librarian.tasks.generate_synthetic.CONFIRM_EXPENSIVE_OPS",
            True,
        ),
        patch(
            "librarian.tasks.generate_synthetic.anthropic.Anthropic",
            return_value=mock_client,
        ),
    ):
        samples = generate_from_chunks(chunks)

    assert samples[0].difficulty == "hard"
