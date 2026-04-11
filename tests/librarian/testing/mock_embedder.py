from __future__ import annotations

import numpy as np


class MockEmbedder:
    """Deterministic fixed-dim embedder for tests — no model load.

    Returns seed-stable random vectors. E5 prefix rule is observed (prefixes are
    stripped before hashing so the same text always returns the same vector).
    """

    def __init__(self, dim: int = 1024, seed: int = 42) -> None:
        self.dim = dim
        self._rng = np.random.default_rng(seed=seed)
        self._cache: dict[str, list[float]] = {}

    def _get(self, text: str) -> list[float]:
        if text not in self._cache:
            vec = self._rng.random(self.dim).astype(float).tolist()
            self._cache[text] = vec
        return self._cache[text]

    def embed_query(self, text: str) -> list[float]:
        # Strip E5 prefix if already present; normalise for cache key
        key = text.removeprefix("query: ")
        return self._get(key)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_passage(self, text: str) -> list[float]:
        key = text.removeprefix("passage: ")
        return self._get(key)

    async def aembed_passage(self, text: str) -> list[float]:
        return self.embed_passage(text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_passage(t) for t in texts]

    async def aembed_passages(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)
