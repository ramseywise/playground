from __future__ import annotations


from librarian.ingestion.base import Chunker, ChunkerConfig
from librarian.ingestion.chunking.html_aware import HtmlAwareChunker, _make_doc_id
from librarian.ingestion.chunking.parent_doc import ParentDocChunker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LONG_SENTENCE = "word " * 60  # 60 words — well above default min_tokens=50


def _doc(text: str, **kwargs: object) -> dict:
    return {
        "url": "https://example.com/doc",
        "title": "Test Doc",
        "text": text,
        **kwargs,
    }


# ---------------------------------------------------------------------------
# ChunkerConfig
# ---------------------------------------------------------------------------


def test_chunker_config_defaults() -> None:
    cfg = ChunkerConfig()
    assert cfg.max_tokens == 512
    assert cfg.min_tokens == 50


# ---------------------------------------------------------------------------
# HtmlAwareChunker — basic
# ---------------------------------------------------------------------------


def test_html_aware_single_section_no_heading() -> None:
    chunker = HtmlAwareChunker(ChunkerConfig(max_tokens=512, min_tokens=5))
    chunks = chunker.chunk_document(_doc(LONG_SENTENCE))
    assert len(chunks) >= 1
    assert all(c.metadata.url == "https://example.com/doc" for c in chunks)


def test_html_aware_heading_split() -> None:
    text = f"## Introduction\n{LONG_SENTENCE}\n\n## Details\n{LONG_SENTENCE}"
    chunker = HtmlAwareChunker(ChunkerConfig(max_tokens=512, min_tokens=5))
    chunks = chunker.chunk_document(_doc(text))
    sections = {c.metadata.section for c in chunks}
    assert "Introduction" in sections
    assert "Details" in sections


def test_html_aware_drops_short_chunks() -> None:
    text = "## Short\ntiny\n\n## Long\n" + LONG_SENTENCE
    chunker = HtmlAwareChunker(ChunkerConfig(max_tokens=512, min_tokens=50))
    chunks = chunker.chunk_document(_doc(text))
    # "tiny" is only 1 word — must be dropped
    for c in chunks:
        assert len(c.text.split()) >= 50


def test_html_aware_full_text_field() -> None:
    doc = {"url": "https://x.com", "title": "T", "full_text": LONG_SENTENCE}
    chunker = HtmlAwareChunker(ChunkerConfig(min_tokens=5))
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1


def test_html_aware_text_field_takes_precedence() -> None:
    doc = {
        "url": "https://x.com",
        "title": "T",
        "text": LONG_SENTENCE,
        "full_text": "should be ignored",
    }
    chunker = HtmlAwareChunker(ChunkerConfig(min_tokens=5))
    chunks = chunker.chunk_document(doc)
    assert all("should be ignored" not in c.text for c in chunks)


def test_html_aware_doc_id_is_deterministic() -> None:
    id1 = _make_doc_id("https://example.com", "Intro")
    id2 = _make_doc_id("https://example.com", "Intro")
    assert id1 == id2
    assert len(id1) == 16


def test_html_aware_doc_id_differs_by_section() -> None:
    assert _make_doc_id("https://x.com", "A") != _make_doc_id("https://x.com", "B")


def test_html_aware_extra_metadata_passthrough() -> None:
    doc = _doc(LONG_SENTENCE, namespace="docs", topic="auth", access_tier="public")
    chunker = HtmlAwareChunker(ChunkerConfig(min_tokens=5))
    chunks = chunker.chunk_document(doc)
    assert all(c.metadata.namespace == "docs" for c in chunks)
    assert all(c.metadata.topic == "auth" for c in chunks)


def test_html_aware_large_section_splits_into_multiple_chunks() -> None:
    long_body = "word " * 600  # 600 words >> max_tokens=512
    text = f"## BigSection\n{long_body}"
    chunker = HtmlAwareChunker(ChunkerConfig(max_tokens=100, min_tokens=5))
    chunks = chunker.chunk_document(_doc(text))
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 110 for c in chunks)  # small slack for overlap


# ---------------------------------------------------------------------------
# ParentDocChunker
# ---------------------------------------------------------------------------


def test_parent_doc_children_have_parent_id() -> None:
    text = f"## Section\n{'word ' * 200}"
    chunker = ParentDocChunker(
        child_config=ChunkerConfig(max_tokens=50, overlap_tokens=10, min_tokens=5)
    )
    chunks = chunker.chunk_document(_doc(text))
    assert all(c.metadata.parent_id is not None for c in chunks)


def test_parent_doc_child_ids_follow_pattern() -> None:
    text = f"## Section\n{'word ' * 200}"
    chunker = ParentDocChunker(
        child_config=ChunkerConfig(max_tokens=50, overlap_tokens=10, min_tokens=5)
    )
    chunks = chunker.chunk_document(_doc(text))
    for c in chunks:
        assert "_child" in c.id


def test_parent_doc_children_share_parent_id_within_section() -> None:
    text = f"## OnlySection\n{'word ' * 200}"
    chunker = ParentDocChunker(
        child_config=ChunkerConfig(max_tokens=50, overlap_tokens=10, min_tokens=5)
    )
    chunks = chunker.chunk_document(_doc(text))
    parent_ids = {c.metadata.parent_id for c in chunks}
    # All children from a single section share one parent_id
    assert len(parent_ids) == 1


def test_parent_doc_multiple_sections_have_distinct_parent_ids() -> None:
    body = "word " * 200
    text = f"## SectionA\n{body}\n\n## SectionB\n{body}"
    chunker = ParentDocChunker(
        child_config=ChunkerConfig(max_tokens=50, overlap_tokens=10, min_tokens=5)
    )
    chunks = chunker.chunk_document(_doc(text))
    parent_ids = {c.metadata.parent_id for c in chunks}
    assert len(parent_ids) == 2


def test_parent_doc_drops_short_children() -> None:
    text = "## Sec\ntiny\n\n## Long\n" + "word " * 200
    chunker = ParentDocChunker(
        child_config=ChunkerConfig(max_tokens=50, overlap_tokens=5, min_tokens=20)
    )
    chunks = chunker.chunk_document(_doc(text))
    for c in chunks:
        assert len(c.text.split()) >= 20


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_html_aware_satisfies_chunker_protocol() -> None:
    assert isinstance(HtmlAwareChunker(), Chunker)


def test_parent_doc_satisfies_chunker_protocol() -> None:
    assert isinstance(ParentDocChunker(), Chunker)
