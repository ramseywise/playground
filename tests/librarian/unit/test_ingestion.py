"""Unit tests for the ingestion pipeline and markdown loaders."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agents.librarian.pipeline.ingestion.loaders import load_directory, load_markdown_file
from agents.librarian.pipeline.ingestion.pipeline import IngestionPipeline, IngestionResult, _sha256
from agents.librarian.tools.storage.vectordb.inmemory import InMemoryRetriever
from tests.librarian.testing.mock_embedder import MockEmbedder
from agents.librarian.tools.storage.metadata_db import MetadataDB
from agents.librarian.tools.storage.snippet_db import SnippetDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_embedder() -> MockEmbedder:
    return MockEmbedder(dim=16)  # tiny dim for speed


@pytest.fixture()
def inmemory_retriever() -> InMemoryRetriever:
    return InMemoryRetriever()


@pytest.fixture()
def meta_db() -> MetadataDB:
    db = MetadataDB(":memory:")
    yield db
    db.close()


@pytest.fixture()
def snippet_db() -> SnippetDB:
    db = SnippetDB(":memory:")
    yield db
    db.close()


@pytest.fixture()
def pipeline(
    mock_embedder: MockEmbedder,
    inmemory_retriever: InMemoryRetriever,
    meta_db: MetadataDB,
    snippet_db: SnippetDB,
) -> IngestionPipeline:
    from agents.librarian.pipeline.ingestion.chunking.strategies import FixedChunker

    return IngestionPipeline(
        chunker=FixedChunker(),
        embedder=mock_embedder,
        vector_store=inmemory_retriever,
        metadata_db=meta_db,
        snippet_db=snippet_db,
        batch_size=8,
    )


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


def test_load_markdown_file_with_frontmatter(tmp_path: Path) -> None:
    md = tmp_path / "test.md"
    md.write_text(
        "---\ntitle: \"Blues History\"\ntopic: blues\ncontent_type: music_history\nsource: blog\n---\n\nBody text here."
    )
    doc = load_markdown_file(md)
    assert doc["title"] == "Blues History"
    assert doc["topic"] == "blues"
    assert doc["content_type"] == "music_history"
    assert doc["text"] == "Body text here."
    assert doc["source_file"] == str(md)


def test_load_markdown_file_no_frontmatter(tmp_path: Path) -> None:
    md = tmp_path / "bare.md"
    md.write_text("Just plain text, no frontmatter.")
    doc = load_markdown_file(md)
    assert doc["text"] == "Just plain text, no frontmatter."
    assert doc["title"] == "Bare"  # derived from filename stem


def test_load_directory_sorted(tmp_path: Path) -> None:
    (tmp_path / "b_article.md").write_text("---\ntitle: B\n---\nContent B.")
    (tmp_path / "a_article.md").write_text("---\ntitle: A\n---\nContent A.")
    docs = load_directory(tmp_path)
    assert len(docs) == 2
    # Should be alphabetically sorted
    assert docs[0]["title"] == "A"
    assert docs[1]["title"] == "B"


def test_load_directory_empty(tmp_path: Path) -> None:
    assert load_directory(tmp_path) == []


def test_load_markdown_null_url(tmp_path: Path) -> None:
    md = tmp_path / "test.md"
    md.write_text("---\nurl: null\n---\nBody.")
    doc = load_markdown_file(md)
    assert doc["url"] == ""


# ---------------------------------------------------------------------------
# Snippet extraction
# ---------------------------------------------------------------------------


def test_extract_snippets_basic() -> None:
    text = (
        "Robert Johnson is a legendary blues musician. "
        "He died at age 27. "
        "His recordings influenced countless artists."
    )
    snippets = IngestionPipeline._extract_snippets(text)
    assert len(snippets) >= 1
    assert all(len(s) >= 30 for s in snippets)


def test_extract_snippets_strips_headings() -> None:
    text = "# My Heading\n\nFirst sentence. Second sentence."
    snippets = IngestionPipeline._extract_snippets(text)
    heading_snippets = [s for s in snippets if s.startswith("#")]
    assert heading_snippets == []


def test_extract_snippets_filters_too_short() -> None:
    text = "Short. " * 20 + "This is a long enough sentence to pass the filter."
    snippets = IngestionPipeline._extract_snippets(text, min_len=30)
    assert all(len(s) >= 30 for s in snippets)


# ---------------------------------------------------------------------------
# IngestionPipeline — ingest_document
# ---------------------------------------------------------------------------


_SAMPLE_DOC = {
    "text": (
        "Robert Johnson is widely considered the first Delta blues musician. "
        "He recorded 29 songs between 1936 and 1937, including Cross Road Blues. "
        "Johnson died at age 27 under mysterious circumstances. "
        "His complex fingerpicking influenced generations of guitarists."
    ),
    "title": "Test Blues Article",
    "url": "",
    "source": "blog",
    "content_type": "music_history",
    "topic": "blues",
    "source_file": "test_blues.md",
}


def test_ingest_document_basic(pipeline: IngestionPipeline, meta_db: MetadataDB, snippet_db: SnippetDB) -> None:
    result = asyncio.run(pipeline.ingest_document(_SAMPLE_DOC))
    assert isinstance(result, IngestionResult)
    assert result.skipped is False
    assert result.doc_id != ""
    assert result.chunk_count >= 1
    assert result.snippet_count >= 1

    # Metadata was written
    doc = meta_db.get_document(result.doc_id)
    assert doc is not None
    assert doc["title"] == "Test Blues Article"
    assert doc["chunk_count"] == result.chunk_count

    # Snippets were written
    snippets = snippet_db.get_snippets_by_doc(result.doc_id)
    assert len(snippets) == result.snippet_count


def test_ingest_document_idempotent(pipeline: IngestionPipeline) -> None:
    """Second ingest of the same document should be skipped."""
    result1 = asyncio.run(pipeline.ingest_document(_SAMPLE_DOC))
    result2 = asyncio.run(pipeline.ingest_document(_SAMPLE_DOC))
    assert result1.skipped is False
    assert result2.skipped is True


def test_ingest_document_empty_text(pipeline: IngestionPipeline) -> None:
    doc = {**_SAMPLE_DOC, "text": ""}
    result = asyncio.run(pipeline.ingest_document(doc))
    assert result.skipped is True


def test_ingest_documents_multiple(pipeline: IngestionPipeline) -> None:
    docs = [
        {**_SAMPLE_DOC, "text": "Blues article body text with enough content.", "source_file": "a.md"},
        {**_SAMPLE_DOC, "text": "Hip-hop article body text with enough content.", "source_file": "b.md"},
    ]
    results = asyncio.run(pipeline.ingest_documents(docs))
    assert len(results) == 2
    assert all(not r.skipped for r in results)


def test_ingest_file(pipeline: IngestionPipeline, tmp_path: Path) -> None:
    md = tmp_path / "article.md"
    md.write_text(
        "---\ntitle: Test Article\ntopic: blues\ncontent_type: music_history\nsource: blog\n---\n\n"
        "Robert Johnson was a Delta blues legend who recorded 29 songs. "
        "His recordings influenced Muddy Waters and many others. "
        "He died young under mysterious circumstances."
    )
    result = asyncio.run(pipeline.ingest_file(md))
    assert result.skipped is False
    assert result.chunk_count >= 1


def test_ingest_directory(pipeline: IngestionPipeline, tmp_path: Path) -> None:
    body = (
        "Robert Johnson was a Delta blues legend. "
        "His recordings set the template for blues guitar. "
        "He died in 1938 at age 27."
    )
    (tmp_path / "blues.md").write_text(f"---\ntitle: Blues\ntopic: blues\ncontent_type: music_history\nsource: blog\n---\n\n{body}")
    (tmp_path / "jazz.md").write_text(f"---\ntitle: Jazz\ntopic: jazz\ncontent_type: music_history\nsource: blog\n---\n\n{body} Jazz is different.")
    results = asyncio.run(pipeline.ingest_directory(tmp_path))
    assert len(results) == 2
    assert all(not r.skipped for r in results)


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------


def test_sha256_deterministic() -> None:
    assert _sha256("hello") == _sha256("hello")
    assert _sha256("hello") != _sha256("world")
