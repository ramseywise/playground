"""Unit tests for RetrievalCache — TTL, LRU eviction, thread safety."""

from __future__ import annotations

import threading
import time

import pytest

from rag.retrieval.cache import RetrievalCache
from rag.schemas.chunks import Chunk, ChunkMetadata
from rag.schemas.retrieval import RetrievalResult


def _make_result(text: str = "chunk text", score: float = 0.9) -> RetrievalResult:
    meta = ChunkMetadata(url="https://example.com")
    chunk = Chunk(id="abc123", text=text, metadata=meta)
    return RetrievalResult(chunk=chunk, score=score, source="test")


class TestRetrievalCacheBasics:
    def test_miss_on_empty_cache(self) -> None:
        cache = RetrievalCache()
        assert cache.get("query", "dense", 5) is None

    def test_put_then_get_returns_results(self) -> None:
        cache = RetrievalCache()
        results = [_make_result()]
        cache.put("hello", "dense", 5, results)
        retrieved = cache.get("hello", "dense", 5)
        assert retrieved is not None
        assert len(retrieved) == 1
        assert retrieved[0].score == pytest.approx(0.9)

    def test_different_strategy_is_cache_miss(self) -> None:
        cache = RetrievalCache()
        cache.put("query", "dense", 5, [_make_result()])
        assert cache.get("query", "sparse", 5) is None

    def test_different_top_k_is_cache_miss(self) -> None:
        cache = RetrievalCache()
        cache.put("query", "dense", 5, [_make_result()])
        assert cache.get("query", "dense", 10) is None

    def test_query_key_normalized_to_lowercase_stripped(self) -> None:
        cache = RetrievalCache()
        cache.put("  Hello  ", "dense", 5, [_make_result()])
        assert cache.get("hello", "dense", 5) is not None

    def test_get_returns_copy_not_reference(self) -> None:
        cache = RetrievalCache()
        results = [_make_result()]
        cache.put("q", "dense", 5, results)
        first = cache.get("q", "dense", 5)
        assert first is not None
        first.clear()
        second = cache.get("q", "dense", 5)
        assert second is not None
        assert len(second) == 1

    def test_clear_empties_cache(self) -> None:
        cache = RetrievalCache()
        cache.put("q", "dense", 5, [_make_result()])
        cache.clear()
        assert cache.get("q", "dense", 5) is None


class TestRetrievalCacheTTL:
    def test_expired_entry_returns_none(self) -> None:
        cache = RetrievalCache(ttl_seconds=0)
        cache.put("q", "dense", 5, [_make_result()])
        # TTL of 0 means expires_at = now + 0 which is immediately ≤ now on next check
        time.sleep(0.01)
        assert cache.get("q", "dense", 5) is None

    def test_fresh_entry_within_ttl_is_returned(self) -> None:
        cache = RetrievalCache(ttl_seconds=60)
        cache.put("q", "dense", 5, [_make_result()])
        assert cache.get("q", "dense", 5) is not None


class TestRetrievalCacheLRUEviction:
    def test_oldest_entry_evicted_when_full(self) -> None:
        cache = RetrievalCache(max_size=2, ttl_seconds=300)
        cache.put("first", "dense", 5, [_make_result("first")])
        cache.put("second", "dense", 5, [_make_result("second")])
        # Adding a third entry should evict "first" (LRU)
        cache.put("third", "dense", 5, [_make_result("third")])
        assert cache.get("first", "dense", 5) is None
        assert cache.get("second", "dense", 5) is not None
        assert cache.get("third", "dense", 5) is not None

    def test_get_promotes_entry_preventing_eviction(self) -> None:
        cache = RetrievalCache(max_size=2, ttl_seconds=300)
        cache.put("alpha", "dense", 5, [_make_result("alpha")])
        cache.put("beta", "dense", 5, [_make_result("beta")])
        # Access "alpha" to promote it to MRU position
        cache.get("alpha", "dense", 5)
        # Now "beta" is LRU; adding "gamma" should evict "beta"
        cache.put("gamma", "dense", 5, [_make_result("gamma")])
        assert cache.get("beta", "dense", 5) is None
        assert cache.get("alpha", "dense", 5) is not None


class TestRetrievalCacheThreadSafety:
    def test_concurrent_writes_dont_corrupt_cache(self) -> None:
        cache = RetrievalCache(max_size=50, ttl_seconds=300)
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                cache.put(f"query_{i}", "dense", 5, [_make_result(f"text_{i}")])
                cache.get(f"query_{i}", "dense", 5)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
