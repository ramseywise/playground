from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agents.researcher.chunker import (
    MAX_CHUNK_PAGES,
    Chunk,
    _hard_split,
    _parse_toc,
    _sections_to_chunks,
    plan_chunks,
)

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


# --- Unit tests ---


def test_single_chunk_when_under_limit() -> None:
    chunks = plan_chunks(Path("fake.pdf"), page_count=12)
    assert len(chunks) == 1
    assert chunks[0].start_page == 1
    assert chunks[0].end_page == 12
    assert chunks[0].title == "Full Document"


def test_single_chunk_at_exact_limit() -> None:
    chunks = plan_chunks(Path("fake.pdf"), page_count=MAX_CHUNK_PAGES)
    assert len(chunks) == 1
    assert chunks[0].end_page == MAX_CHUNK_PAGES


def test_hard_split_fallback_when_no_toc() -> None:
    with patch("agents.researcher.chunker.extract_toc", return_value="No headings here."):
        chunks = plan_chunks(Path("fake.pdf"), page_count=45)
    assert all(c.end_page - c.start_page + 1 <= MAX_CHUNK_PAGES for c in chunks)
    assert chunks[0].start_page == 1
    assert chunks[-1].end_page == 45
    assert all("Part" in c.title for c in chunks)


def test_hard_split_coverage() -> None:
    chunks = _hard_split(page_count=55)
    starts = [c.start_page for c in chunks]
    ends = [c.end_page for c in chunks]
    assert starts[0] == 1
    assert ends[-1] == 55
    assert all(e - s + 1 <= MAX_CHUNK_PAGES for s, e in zip(starts, ends))
    # No gaps between chunks
    for i in range(len(chunks) - 1):
        assert chunks[i].end_page + 1 == chunks[i + 1].start_page


def test_toc_detected_chapters() -> None:
    toc_text = (
        "Chapter 1: Introduction  3\n"
        "Chapter 2: Background    25\n"
        "Chapter 3: Methods       45\n"
    )
    with patch("agents.researcher.chunker.extract_toc", return_value=toc_text):
        chunks = plan_chunks(Path("fake.pdf"), page_count=60)
    assert all(c.end_page - c.start_page + 1 <= MAX_CHUNK_PAGES for c in chunks)
    # Chapter 1 and 2 span 22 and 20 pages respectively — chapter 1 oversized → sub-split
    titles = [c.title for c in chunks]
    assert any("Introduction" in t for t in titles)
    assert any("Background" in t for t in titles)
    assert any("Methods" in t for t in titles)


def test_oversized_chapter_is_sub_split() -> None:
    sections = [("Chapter 1: Big Chapter", 1), ("Chapter 2: Small", 50)]
    chunks = _sections_to_chunks(sections, page_count=60)
    for c in chunks:
        assert c.end_page - c.start_page + 1 <= MAX_CHUNK_PAGES
    assert any("Part 1" in c.title for c in chunks)
    assert any("Part 2" in c.title for c in chunks)


def test_parse_toc_empty_when_no_match() -> None:
    sections = _parse_toc("Random text without headings.", total_pages=50)
    assert sections == []


def test_parse_toc_extracts_chapter_pages() -> None:
    toc_text = "Chapter 1: Intro   5\nChapter 2: Core   20\n"
    sections = _parse_toc(toc_text, total_pages=50)
    assert len(sections) == 2
    assert sections[0] == ("Chapter 1: Intro", 5)
    assert sections[1] == ("Chapter 2: Core", 20)


def test_parse_toc_ignores_pages_out_of_range() -> None:
    toc_text = "Chapter 1: Intro   5\nChapter 2: Core   999\n"
    sections = _parse_toc(toc_text, total_pages=50)
    assert len(sections) == 1
    assert sections[0][1] == 5


def test_chunk_model_is_pydantic() -> None:
    c = Chunk(start_page=1, end_page=20, title="Intro")
    assert c.start_page == 1
    assert c.model_dump()["title"] == "Intro"


# --- Integration test ---


@pytest.mark.integration
def test_plan_chunks_real_pdf() -> None:
    """sample.pdf is 12 pages — should return a single chunk."""
    from agents.researcher.chunker import plan_chunks

    chunks = plan_chunks(FIXTURE_PDF, page_count=12)
    assert len(chunks) == 1
    assert chunks[0].start_page == 1
    assert chunks[0].end_page == 12
