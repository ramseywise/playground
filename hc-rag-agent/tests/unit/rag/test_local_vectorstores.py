"""Tests for local vector indexes (mocked embeddings; optional Chroma/DuckDB)."""

from __future__ import annotations

from pathlib import Path
import pytest

from rag.schemas.chunks import Chunk, ChunkMetadata
from rag.datastore import DictVectorIndex


class _FakeEmbeddings:
    dim = 4

    def embed_query(self, text: str) -> list[float]:
        _ = text
        return [1.0, 0.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i % 2), 1.0, 0.0, 0.0] for i, _ in enumerate(texts)]


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.rag.datastore.local.get_embeddings",
        lambda: _FakeEmbeddings(),
    )


@pytest.mark.asyncio
async def test_dict_index_upsert_and_search(
    fake_embeddings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = DictVectorIndex()
    chunks = [
        Chunk(
            id="a",
            text="hello world",
            embedding=[1.0, 0.0, 0.0, 0.0],
            metadata=ChunkMetadata(url="https://a.example"),
        ),
        Chunk(
            id="b",
            text="other",
            embedding=[0.0, 1.0, 0.0, 0.0],
            metadata=ChunkMetadata(url="https://b.example"),
        ),
    ]
    await idx.upsert(chunks)
    q = [1.0, 0.0, 0.0, 0.0]
    results = await idx.search("ignored", q, k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_dict_index_similarity_search_with_score(fake_embeddings: None) -> None:
    idx = DictVectorIndex()
    idx.upsert_flat(
        [
            (
                "id1",
                "alpha",
                {"source": "s1"},
                [1.0, 0.0, 0.0, 0.0],
            ),
        ],
    )
    hits = idx.similarity_search_with_score("q", k=1)
    assert len(hits) == 1
    doc, score = hits[0]
    assert "alpha" in doc.page_content
    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_chroma_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("chromadb")

    monkeypatch.setattr(
        "src.rag.datastore.local.get_embeddings",
        lambda: _FakeEmbeddings(),
    )

    from rag.datastore import ChromaVectorIndex

    idx = ChromaVectorIndex(tmp_path / "chroma_data")
    await idx.upsert(
        [
            Chunk(
                id="c1",
                text="persisted",
                embedding=[0.0, 0.0, 1.0, 0.0],
                metadata=ChunkMetadata(doc_id="c1"),
            ),
        ],
    )
    assert idx.doc_count() == 1
    hits = idx.similarity_search_with_score("q", k=2)
    assert len(hits) == 1


@pytest.mark.asyncio
async def test_duckdb_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("duckdb")

    monkeypatch.setattr(
        "src.rag.datastore.local.get_embeddings",
        lambda: _FakeEmbeddings(),
    )

    from rag.datastore import DuckDBVectorIndex

    db = tmp_path / "t.duckdb"
    idx = DuckDBVectorIndex(db)
    await idx.upsert(
        [
            Chunk(
                id="d1",
                text="duck row",
                embedding=[0.0, 1.0, 0.0, 0.0],
                metadata=ChunkMetadata(title="t"),
            ),
        ],
    )
    assert idx.doc_count() == 1
    rows = idx.similarity_search_with_score("q", k=3)
    assert len(rows) == 1


def test_vectorstore_factory_memory_mocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from rag.datastore import factory as vs
    from rag.datastore import factory as rt

    monkeypatch.setenv("VECTOR_STORE_BACKEND", "memory")
    monkeypatch.setattr(rt, "_persist_base", lambda: tmp_path)
    monkeypatch.setattr(
        "src.rag.datastore.factory.get_embeddings", lambda: _FakeEmbeddings()
    )
    vs.reset_vectorstore_for_tests()
    monkeypatch.setattr(rt, "_project_data_dir", lambda: tmp_path / "proj_data")
    (tmp_path / "proj_data" / "embeddings").mkdir(parents=True)
    (tmp_path / "proj_data" / "document").mkdir(parents=True)
    (tmp_path / "proj_data" / "document" / "a.txt").write_text(
        "hello", encoding="utf-8"
    )

    store = vs.get_vectorstore()
    assert store.doc_count() >= 1
    vs.reset_vectorstore_for_tests()
