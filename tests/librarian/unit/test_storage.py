"""Unit tests for MetadataDB and SnippetDB.

Both classes use DuckDB ':memory:' paths so no files are created on disk.
"""

from __future__ import annotations

import pytest

from storage.metadata_db import MetadataDB
from storage.snippet_db import SnippetDB


# ---------------------------------------------------------------------------
# MetadataDB
# ---------------------------------------------------------------------------


@pytest.fixture()
def meta_db() -> MetadataDB:
    db = MetadataDB(":memory:")
    yield db
    db.close()


def _insert_doc(db: MetadataDB, *, doc_id: str = "doc1", checksum: str = "abc123") -> None:
    db.insert_document(
        doc_id,
        title="Blues History",
        source="blog",
        source_file="data/raw/blues_history.md",
        content_type="music_history",
        topic="blues",
        word_count=500,
        chunk_count=10,
        snippet_count=25,
        checksum=checksum,
    )


class TestMetadataDB:
    def test_insert_and_get(self, meta_db: MetadataDB) -> None:
        _insert_doc(meta_db)
        doc = meta_db.get_document("doc1")
        assert doc is not None
        assert doc["title"] == "Blues History"
        assert doc["topic"] == "blues"
        assert doc["chunk_count"] == 10
        assert doc["snippet_count"] == 25

    def test_get_missing_returns_none(self, meta_db: MetadataDB) -> None:
        assert meta_db.get_document("nonexistent") is None

    def test_document_exists_by_checksum_true(self, meta_db: MetadataDB) -> None:
        _insert_doc(meta_db, checksum="deadbeef")
        assert meta_db.document_exists_by_checksum("deadbeef") is True

    def test_document_exists_by_checksum_false(self, meta_db: MetadataDB) -> None:
        assert meta_db.document_exists_by_checksum("nothere") is False

    def test_checksum_uniqueness_raises(self, meta_db: MetadataDB) -> None:
        """Inserting two docs with the same checksum should raise (UNIQUE constraint)."""
        _insert_doc(meta_db, doc_id="doc1", checksum="same")
        with pytest.raises(Exception):  # noqa: B017
            _insert_doc(meta_db, doc_id="doc2", checksum="same")

    def test_list_documents_empty(self, meta_db: MetadataDB) -> None:
        assert meta_db.list_documents() == []

    def test_list_documents_multiple(self, meta_db: MetadataDB) -> None:
        _insert_doc(meta_db, doc_id="doc1", checksum="cs1")
        _insert_doc(meta_db, doc_id="doc2", checksum="cs2")
        docs = meta_db.list_documents()
        assert len(docs) == 2
        ids = {d["doc_id"] for d in docs}
        assert ids == {"doc1", "doc2"}

    def test_context_manager(self) -> None:
        with MetadataDB(":memory:") as db:
            _insert_doc(db)
            assert db.get_document("doc1") is not None


# ---------------------------------------------------------------------------
# SnippetDB
# ---------------------------------------------------------------------------


@pytest.fixture()
def snippet_db() -> SnippetDB:
    db = SnippetDB(":memory:")
    yield db
    db.close()


def _make_snippets(doc_id: str = "doc1") -> list[dict]:
    return [
        {
            "id": f"{doc_id}_0",
            "doc_id": doc_id,
            "text": "Robert Johnson is widely considered the first Delta blues musician.",
            "title": "Blues History",
            "topic": "blues",
            "position": 0,
            "source": "blog",
        },
        {
            "id": f"{doc_id}_1",
            "doc_id": doc_id,
            "text": "Muddy Waters moved from Mississippi to Chicago in 1943.",
            "title": "Blues History",
            "topic": "blues",
            "position": 1,
            "source": "blog",
        },
        {
            "id": f"{doc_id}_2",
            "doc_id": doc_id,
            "text": "B.B. King named his guitar Lucille after a woman at a dance hall.",
            "title": "Blues History",
            "topic": "blues",
            "position": 2,
            "source": "blog",
        },
    ]


class TestSnippetDB:
    def test_insert_and_get_by_doc(self, snippet_db: SnippetDB) -> None:
        snippet_db.insert_snippets(_make_snippets())
        rows = snippet_db.get_snippets_by_doc("doc1")
        assert len(rows) == 3
        # Ordered by position
        assert rows[0]["position"] == 0
        assert rows[1]["position"] == 1
        assert rows[2]["position"] == 2

    def test_get_snippets_unknown_doc(self, snippet_db: SnippetDB) -> None:
        assert snippet_db.get_snippets_by_doc("unknown") == []

    def test_search_snippets_ilike(self, snippet_db: SnippetDB) -> None:
        """ILIKE fallback search should find relevant snippet."""
        snippet_db.insert_snippets(_make_snippets())
        results = snippet_db.search_snippets("Muddy Waters", k=5)
        assert len(results) >= 1
        texts = [r["text"] for r in results]
        assert any("Muddy Waters" in t for t in texts)

    def test_search_snippets_returns_score(self, snippet_db: SnippetDB) -> None:
        snippet_db.insert_snippets(_make_snippets())
        results = snippet_db.search_snippets("Robert Johnson", k=5)
        assert all("score" in r for r in results)

    def test_search_snippets_no_match_returns_empty(self, snippet_db: SnippetDB) -> None:
        snippet_db.insert_snippets(_make_snippets())
        results = snippet_db.search_snippets("xyzzy_no_match_term", k=5)
        assert results == []

    def test_insert_idempotent(self, snippet_db: SnippetDB) -> None:
        """Inserting the same snippets twice should not raise (INSERT OR IGNORE)."""
        snippet_db.insert_snippets(_make_snippets())
        snippet_db.insert_snippets(_make_snippets())  # should not raise
        rows = snippet_db.get_snippets_by_doc("doc1")
        assert len(rows) == 3  # still 3, not 6

    def test_context_manager(self) -> None:
        with SnippetDB(":memory:") as db:
            db.insert_snippets(_make_snippets())
            assert len(db.get_snippets_by_doc("doc1")) == 3
