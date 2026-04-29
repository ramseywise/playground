"""DuckDB ingestion sidecar tables (metadata + snippets)."""

from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from rag.datastore import (  # noqa: E402 — after importorskip when duckdb optional
    DuckDBDocumentMetadataStore,
    DuckDBSnippetStore,
)


def test_metadata_checksum_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "t.duckdb"
    meta = DuckDBDocumentMetadataStore(db)
    assert meta.document_exists_by_checksum("abc") is False
    meta.insert_document(
        "doc1",
        checksum="abc",
        title="T",
        source="s",
        source_file="f.md",
        content_type="md",
        topic="x",
        word_count=10,
        chunk_count=2,
        snippet_count=1,
    )
    assert meta.document_exists_by_checksum("abc") is True


def test_snippet_insert_and_search(tmp_path: Path) -> None:
    db = tmp_path / "t.duckdb"
    sn = DuckDBSnippetStore(db)
    sn.insert_snippets(
        [
            {
                "id": "s1",
                "doc_id": "d1",
                "text": "hello world from support",
                "title": "T",
                "topic": "",
                "position": 0,
                "source": "src",
            },
        ],
    )
    rows = sn.search_snippets("support", k=5)
    assert len(rows) == 1
    assert rows[0]["text"] == "hello world from support"
