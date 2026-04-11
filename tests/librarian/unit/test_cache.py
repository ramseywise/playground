from __future__ import annotations

from librarian.retrieval.cache import RetrievalCache
from librarian.schemas.chunks import Chunk, ChunkMetadata
from librarian.schemas.retrieval import RetrievalResult


def _result(chunk_id: str = "c1", score: float = 0.5) -> RetrievalResult:
    chunk = Chunk(
        id=chunk_id,
        text="text",
        metadata=ChunkMetadata(url="https://example.com", title="T", doc_id="d1"),
    )
    return RetrievalResult(chunk=chunk, score=score, source="hybrid")


def test_cache_hit_miss(monkeypatch) -> None:
    cache = RetrievalCache(max_size=2, ttl_seconds=10)
    monkeypatch.setattr("librarian.retrieval.cache.time.monotonic", lambda: 1.0)
    assert cache.get("q", "inmemory", 5) is None
    cache.put("q", "inmemory", 5, [_result()])
    cached = cache.get("q", "inmemory", 5)
    assert cached is not None
    assert cached[0].chunk.id == "c1"


def test_cache_expiry(monkeypatch) -> None:
    cache = RetrievalCache(max_size=2, ttl_seconds=1)
    times = iter([1.0, 1.0, 2.1])
    monkeypatch.setattr(
        "librarian.retrieval.cache.time.monotonic", lambda: next(times)
    )
    cache.put("q", "inmemory", 5, [_result()])
    assert cache.get("q", "inmemory", 5) is not None
    assert cache.get("q", "inmemory", 5) is None


def test_cache_eviction(monkeypatch) -> None:
    cache = RetrievalCache(max_size=1, ttl_seconds=10)
    monkeypatch.setattr("librarian.retrieval.cache.time.monotonic", lambda: 1.0)
    cache.put("a", "inmemory", 5, [_result("a")])
    cache.put("b", "inmemory", 5, [_result("b")])
    assert cache.get("a", "inmemory", 5) is None
    assert cache.get("b", "inmemory", 5) is not None
