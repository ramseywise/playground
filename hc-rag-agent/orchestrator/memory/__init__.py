"""Episodic memory layer — cross-session recall via LangGraph Store API.

Memory tiers
------------
Working memory   : ``state.messages`` (GraphState, per-turn)
Session memory   : LangGraph checkpointer (thread persistence, already wired)
Episodic memory  : This module — distilled facts that survive across sessions
Semantic memory  : ``app/rag/`` — the indexed knowledge base

Usage
-----
Wire a ``BaseStore`` instance (e.g. ``InMemoryStore`` for dev, a persistent
store for prod) into :class:`EpisodicMemory` and inject it via graph config:

    store = InMemoryStore()
    graph = build_graph(checkpointer, store=store)

Inside a node, read/write via the injected store handle:

    from orchestrator.memory import EpisodicMemory
    mem = EpisodicMemory(store, user_id=state.user_id)
    facts = await mem.recall("billing")
    await mem.remember("billing", {"preferred_contact": "email"})
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Namespace prefix — scopes all keys so multiple agents can share one store.
_NS_PREFIX = ("support_rag", "episodic")


class EpisodicMemory:
    """Thin wrapper around a LangGraph ``BaseStore`` for cross-session user facts.

    Parameters
    ----------
    store:
        A ``langgraph.store.base.BaseStore`` instance (sync or async).
    user_id:
        Identifies the user whose facts are being stored.  Used as the
        innermost namespace segment so facts are user-scoped.
    """

    def __init__(self, store: Any, *, user_id: str) -> None:
        self._store = store
        self._ns = (*_NS_PREFIX, user_id)

    async def remember(self, key: str, value: dict[str, Any]) -> None:
        """Persist a fact under *key* for this user."""
        await self._store.aput(self._ns, key, value)
        log.debug("memory.remember user_ns=%s key=%s", self._ns, key)

    async def recall(self, key: str) -> dict[str, Any] | None:
        """Return the stored fact for *key*, or ``None`` if not found."""
        item = await self._store.aget(self._ns, key)
        if item is None:
            return None
        return item.value  # type: ignore[no-any-return]

    async def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Semantic search over all facts for this user.

        Requires a store that supports vector search (e.g. ``AsyncPostgresStore``
        with an embedding config).  Falls back gracefully to ``[]`` for stores
        that raise ``NotImplementedError``.
        """
        try:
            results = await self._store.asearch(self._ns, query=query, limit=limit)
            return [r.value for r in results]
        except NotImplementedError:
            log.debug("memory.search: store does not support vector search, skipping")
            return []

    async def forget(self, key: str) -> None:
        """Delete a stored fact by key."""
        await self._store.adelete(self._ns, key)
        log.debug("memory.forget user_ns=%s key=%s", self._ns, key)


__all__ = ["EpisodicMemory"]
