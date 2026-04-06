from __future__ import annotations

from pathlib import Path

import pytest

from agents.research.models import (
    NoteMetadata,
    Note,
    SOURCE_TOPIC_MAP,
    resolve_topic,
)


# --- resolve_topic ---


@pytest.mark.parametrize(
    "folder,expected_slug",
    list(SOURCE_TOPIC_MAP.items()),
)
def test_resolve_topic_all_known_folders(
    folder: str, expected_slug: str, tmp_path: Path
) -> None:
    pdf_path = Path("/Users/wiseer/Dropbox/ai_readings") / folder / "some_paper.pdf"
    # Patch OBSIDIAN_TOPICS to avoid creating real dirs during tests
    import agents.research.models as models_module

    original = models_module.OBSIDIAN_TOPICS
    models_module.OBSIDIAN_TOPICS = tmp_path
    try:
        slug = resolve_topic(pdf_path)
        assert slug == expected_slug
        assert (tmp_path / slug).exists()
    finally:
        models_module.OBSIDIAN_TOPICS = original


def test_resolve_topic_knowledge_graphs() -> None:
    """Specifically verify the new knowledge-graphs folder mapping."""
    pdf_path = Path(
        "/Users/wiseer/Dropbox/ai_readings/2.knowledge graphs/chapter13.pdf"
    )
    import agents.research.models as models_module

    from pathlib import Path as _Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = _Path(tmp)
        original = models_module.OBSIDIAN_TOPICS
        models_module.OBSIDIAN_TOPICS = tmp_path
        try:
            slug = resolve_topic(pdf_path)
            assert slug == "knowledge-graphs"
            assert (tmp_path / "knowledge-graphs").exists()
        finally:
            models_module.OBSIDIAN_TOPICS = original


def test_resolve_topic_unknown_folder_raises() -> None:
    pdf_path = Path("/Users/wiseer/Dropbox/ai_readings/unknown_folder/paper.pdf")
    with pytest.raises(ValueError, match="No known source folder found"):
        resolve_topic(pdf_path)


# --- NoteMetadata ---


def _valid_metadata(**overrides: object) -> NoteMetadata:
    defaults = dict(
        title="Test Note",
        source="paper",
        topic="rag",
        tags=["rag", "retrieval"],
        date="2026-04-03",
        relevance=4,
        source_file="0.rag/test_paper.pdf",
        pages="1-12",
    )
    defaults.update(overrides)
    return NoteMetadata(**defaults)  # type: ignore[arg-type]


def test_note_metadata_valid() -> None:
    meta = _valid_metadata()
    assert meta.title == "Test Note"
    assert meta.relevance == 4


def test_note_metadata_relevance_out_of_range_raises() -> None:
    with pytest.raises(Exception):
        _valid_metadata(relevance=6)


def test_note_metadata_relevance_zero_raises() -> None:
    with pytest.raises(Exception):
        _valid_metadata(relevance=0)


def test_note_metadata_invalid_source_raises() -> None:
    with pytest.raises(Exception):
        _valid_metadata(source="blog-post")


def test_note_metadata_all_valid_sources() -> None:
    for src in ("book-chapter", "paper", "course", "article"):
        meta = _valid_metadata(source=src)
        assert meta.source == src


# --- Note ---


def test_note_model() -> None:
    meta = _valid_metadata()
    note = Note(metadata=meta, body="## Summary\nSome summary here.")
    assert note.body.startswith("## Summary")
    assert note.metadata.topic == "rag"
