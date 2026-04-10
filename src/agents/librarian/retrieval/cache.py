from __future__ import annotations

import threading
import time
from collections import OrderedDict

from agents.librarian.schemas.retrieval import RetrievalResult


class RetrievalCache:
    """Thread-safe TTL LRU cache for retrieval results."""

    def __init__(self, max_size: int = 256, ttl_seconds: int = 300) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._entries: OrderedDict[
            tuple[str, str, int], tuple[float, list[RetrievalResult]]
        ] = OrderedDict()

    def get(
        self, query: str, strategy: str, top_k: int
    ) -> list[RetrievalResult] | None:
        key = self._key(query, strategy, top_k)
        now = time.monotonic()
        with self._lock:
            item = self._entries.get(key)
            if item is None:
                return None
            expires_at, results = item
            if expires_at <= now:
                self._entries.pop(key, None)
                return None
            self._entries.move_to_end(key)
            return list(results)

    def put(
        self,
        query: str,
        strategy: str,
        top_k: int,
        results: list[RetrievalResult],
    ) -> None:
        key = self._key(query, strategy, top_k)
        expires_at = time.monotonic() + self._ttl_seconds
        with self._lock:
            self._entries[key] = (expires_at, list(results))
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_size:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    @staticmethod
    def _key(query: str, strategy: str, top_k: int) -> tuple[str, str, int]:
        return query.strip().lower(), strategy, top_k
