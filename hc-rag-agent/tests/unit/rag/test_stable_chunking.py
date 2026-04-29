"""Stable chunk ids (legacy notebook alignment) and corpus_v2 helpers."""

from __future__ import annotations

from pathlib import Path

from rag.preprocessing.base import ChunkerConfig
from rag.preprocessing.chunking.strategies import FixedChunker
from rag.preprocessing.chunking.utils import (
    make_chunk,
    stable_doc_id_from_document,
)
from rag.ingestion.corpus_v2 import _normalize_v2_doc, load_jsonl_corpus


def test_stable_doc_id_help_article() -> None:
    url = "https://hilfe.sevdesk.de/de/articles/10570020-darlehen"
    assert (
        stable_doc_id_from_document({"url": url, "source": "help_center"})
        == "help_10570020"
    )


def test_stable_chunk_ids_fixed_chunker() -> None:
    cfg = ChunkerConfig(
        max_tokens=50,
        min_tokens=1,
        overlap_tokens=0,
        chunk_id_mode="stable",
    )
    ch = FixedChunker(config=cfg)
    text = "word " * 200
    doc = {
        "text": text,
        "url": "https://hilfe.sevdesk.de/de/articles/99-x",
        "title": "t",
        "source": "help_center",
    }
    chunks = ch.chunk_document(doc)
    assert len(chunks) >= 2
    assert chunks[0].id == "help_99_0"
    assert chunks[1].id == "help_99_1"
    assert chunks[0].metadata.doc_id == "help_99"


def test_make_chunk_stable_explicit_id() -> None:
    c = make_chunk(
        "hello",
        "https://x",
        "t",
        "s",
        "help_1",
        chunk_index=0,
        chunk_id_mode="stable",
    )
    assert c.id == "help_1_0"


def test_normalize_v2_doc_maps_content_to_text() -> None:
    d = _normalize_v2_doc(
        {"content": "hello there", "url": "https://hilfe.sevdesk.de/de/articles/5-x"}
    )
    assert d["text"] == "hello there"


def test_load_jsonl_corpus(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text(
        '{"text":"hello world hello world", "url":"https://hilfe.sevdesk.de/de/articles/12-z", "title":"T"}\n',
        encoding="utf-8",
    )
    docs = load_jsonl_corpus(tmp_path)
    assert len(docs) == 1
    assert docs[0]["stable_doc_id"] == "help_12"
