from __future__ import annotations

from typing import Any


class Registry:
    """Strategy registry — maps string keys to implementation classes.

    Populated in Step 4 (retrieval) and Step 5 (reranker).
    """

    _store: dict[str, Any] = {}

    @classmethod
    def register(cls, key: str, impl: Any) -> None:
        cls._store[key] = impl

    @classmethod
    def get(cls, key: str) -> Any:
        if key not in cls._store:
            raise KeyError(f"No strategy registered for key: {key!r}")
        return cls._store[key]

    @classmethod
    def clear(cls) -> None:
        cls._store.clear()
