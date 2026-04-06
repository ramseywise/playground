from __future__ import annotations

from pathlib import Path

import pytest

from agents.research.models import Note, NoteMetadata
from agents.research.writer import render_note, sanitize_filename, write_note


def _make_note(**overrides: object) -> Note:
    defaults = dict(
        title="Knowledge Graph RAG",
        source="book-chapter",
        topic="knowledge-graphs",
        tags=["rag", "knowledge-graph"],
        date="2026-04-03",
        relevance=4,
        source_file="2.knowledge graphs/ch13.pdf",
        pages="1-23",
    )
    defaults.update(overrides)
    meta = NoteMetadata(**defaults)  # type: ignore[arg-type]
    return Note(metadata=meta, body="## Summary\nSome summary here.\n")


# --- sanitize_filename ---


def test_sanitize_simple() -> None:
    assert sanitize_filename("Hello World") == "hello-world.md"


def test_sanitize_special_chars() -> None:
    assert sanitize_filename("Graph–Powered RAG: A Review") == "graphpowered-rag-a-review.md"


def test_sanitize_long_title_truncated() -> None:
    long = "A" * 100
    result = sanitize_filename(long)
    assert len(result) <= 83  # 80 chars + ".md"


def test_sanitize_no_double_dashes() -> None:
    result = sanitize_filename("foo  --  bar")
    assert "--" not in result


# --- render_note ---


def test_render_note_has_frontmatter() -> None:
    note = _make_note()
    rendered = render_note(note)
    assert rendered.startswith("---\n")
    assert "title: Knowledge Graph RAG" in rendered
    assert "relevance: 4" in rendered
    assert "topic: knowledge-graphs" in rendered


def test_render_note_tags_in_frontmatter() -> None:
    note = _make_note(tags=["rag", "embeddings"])
    rendered = render_note(note)
    assert "rag" in rendered
    assert "embeddings" in rendered


def test_render_note_body_follows_frontmatter() -> None:
    note = _make_note()
    rendered = render_note(note)
    assert "## Summary" in rendered
    fm_end = rendered.index("---\n", 4)  # second ---
    body_start = rendered.index("## Summary")
    assert body_start > fm_end


# --- write_note ---


def test_write_note_creates_file(tmp_path: Path) -> None:
    import agents.research.writer as writer_module

    original_topics = writer_module.OBSIDIAN_TOPICS
    original_index = writer_module.OBSIDIAN_INDEX
    writer_module.OBSIDIAN_TOPICS = tmp_path / "topics"
    writer_module.OBSIDIAN_INDEX = tmp_path / "_index.md"

    try:
        note = _make_note()
        note_path = write_note(note, Path("fake.pdf"))
        assert note_path.exists()
        content = note_path.read_text()
        assert "title: Knowledge Graph RAG" in content
        assert "## Summary" in content
    finally:
        writer_module.OBSIDIAN_TOPICS = original_topics
        writer_module.OBSIDIAN_INDEX = original_index


def test_write_note_appends_index(tmp_path: Path) -> None:
    import agents.research.writer as writer_module

    original_topics = writer_module.OBSIDIAN_TOPICS
    original_index = writer_module.OBSIDIAN_INDEX
    writer_module.OBSIDIAN_TOPICS = tmp_path / "topics"
    writer_module.OBSIDIAN_INDEX = tmp_path / "_index.md"

    try:
        note = _make_note()
        write_note(note, Path("fake.pdf"))
        index = (tmp_path / "_index.md").read_text()
        assert "knowledge-graphs" in index
        assert "4/5" in index
        assert "2026-04-03" in index
    finally:
        writer_module.OBSIDIAN_TOPICS = original_topics
        writer_module.OBSIDIAN_INDEX = original_index


def test_write_note_raises_if_exists(tmp_path: Path) -> None:
    import agents.research.writer as writer_module

    original_topics = writer_module.OBSIDIAN_TOPICS
    original_index = writer_module.OBSIDIAN_INDEX
    writer_module.OBSIDIAN_TOPICS = tmp_path / "topics"
    writer_module.OBSIDIAN_INDEX = tmp_path / "_index.md"

    try:
        note = _make_note()
        write_note(note, Path("fake.pdf"))
        with pytest.raises(FileExistsError):
            write_note(note, Path("fake.pdf"))
    finally:
        writer_module.OBSIDIAN_TOPICS = original_topics
        writer_module.OBSIDIAN_INDEX = original_index


def test_index_created_if_missing(tmp_path: Path) -> None:
    import agents.research.writer as writer_module

    original_topics = writer_module.OBSIDIAN_TOPICS
    original_index = writer_module.OBSIDIAN_INDEX
    writer_module.OBSIDIAN_TOPICS = tmp_path / "topics"
    index_path = tmp_path / "_index.md"
    writer_module.OBSIDIAN_INDEX = index_path

    try:
        assert not index_path.exists()
        note = _make_note()
        write_note(note, Path("fake.pdf"))
        assert index_path.exists()
        content = index_path.read_text()
        assert "# Research Index" in content
    finally:
        writer_module.OBSIDIAN_TOPICS = original_topics
        writer_module.OBSIDIAN_INDEX = original_index
