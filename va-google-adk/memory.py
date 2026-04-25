"""Lightweight preference and episodic memory store backed by SQLite.

Schema: preference_store(user_id, key, value, updated_at)

Both user preferences and session summaries share this table:
  - Preferences: key = "pref:<name>",          value = user-chosen value
  - Session summaries: key = "session:<id>",    value = 1-sentence summary
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_DB_PATH = os.getenv("MEMORY_DB_PATH", "memory.db")

_DDL = """
CREATE TABLE IF NOT EXISTS preference_store (
    user_id    TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
)
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _init_sync() -> None:
    with sqlite3.connect(_DB_PATH) as db:
        db.execute(_DDL)
        db.commit()


def _upsert_sync(user_id: str, key: str, value: str) -> None:
    with sqlite3.connect(_DB_PATH) as db:
        db.execute(
            """INSERT INTO preference_store (user_id, key, value, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (user_id, key)
               DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
            (user_id, key, value, _now()),
        )
        db.commit()


def _delete_sync(user_id: str, key: str) -> None:
    with sqlite3.connect(_DB_PATH) as db:
        db.execute(
            "DELETE FROM preference_store WHERE user_id = ? AND key = ?",
            (user_id, key),
        )
        db.commit()


def _get_top_sync(user_id: str, n: int) -> list[dict]:
    with sqlite3.connect(_DB_PATH) as db:
        db.row_factory = sqlite3.Row
        cursor = db.execute(
            """SELECT key, value, updated_at FROM preference_store
               WHERE user_id = ? AND key NOT LIKE 'session:%'
               ORDER BY updated_at DESC
               LIMIT ?""",
            (user_id, n),
        )
        return [dict(r) for r in cursor.fetchall()]


# ── async wrappers ─────────────────────────────────────────────────────────


async def init_memory_db() -> None:
    """Create the preference_store table if it does not exist."""
    await asyncio.to_thread(_init_sync)
    logger.info("Memory DB ready at %s", _DB_PATH)


async def upsert(user_id: str, key: str, value: str) -> None:
    """Insert or update a preference (or session summary) entry."""
    await asyncio.to_thread(_upsert_sync, user_id, key, value)


async def delete(user_id: str, key: str) -> None:
    """Delete a preference entry."""
    await asyncio.to_thread(_delete_sync, user_id, key)


async def get_top(user_id: str, n: int = 3) -> list[dict]:
    """Return the n most recently updated preferences (excluding session summaries)."""
    return await asyncio.to_thread(_get_top_sync, user_id, n)
