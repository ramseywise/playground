"""Artefact store — local filesystem (dev) or S3 (prod).

Environment variables:
  ARTEFACT_BACKEND    local | s3            (default: local)
  ARTEFACT_LOCAL_DIR  ./artefacts           (local backend root)
  ARTEFACT_S3_BUCKET  <bucket>              (s3 backend)
  ARTEFACT_TTL_DAYS   30                    (retention, days)
  GATEWAY_BASE_URL    http://localhost:8000  (for local download URLs)
  MEMORY_DB_PATH      memory.db             (metadata DB, shared with preference_store)
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

_DB_PATH = os.getenv("MEMORY_DB_PATH", "memory.db")
_BACKEND = os.getenv("ARTEFACT_BACKEND", "local")
_LOCAL_DIR = Path(os.getenv("ARTEFACT_LOCAL_DIR", "./artefacts"))
_S3_BUCKET = os.getenv("ARTEFACT_S3_BUCKET", "")
_DEFAULT_TTL = int(os.getenv("ARTEFACT_TTL_DAYS", "30"))
_BASE_URL = os.getenv("GATEWAY_BASE_URL", "http://localhost:8000")

_DDL = """
CREATE TABLE IF NOT EXISTS artefacts (
    artefact_id  TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    filename     TEXT NOT NULL,
    storage_key  TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'text/markdown',
    created_at   TEXT NOT NULL,
    ttl_days     INTEGER NOT NULL DEFAULT 30,
    deleted_at   TEXT
)"""


# ---------------------------------------------------------------------------
# DB helpers (stdlib sqlite3 + asyncio.to_thread)
# ---------------------------------------------------------------------------


def _init_sync(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(_DDL)
        conn.commit()


def _insert_sync(db_path: str, row: dict) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO artefacts "
            "(artefact_id, session_id, filename, storage_key, content_type, created_at, ttl_days) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                row["artefact_id"], row["session_id"], row["filename"],
                row["storage_key"], row["content_type"], row["created_at"],
                row["ttl_days"],
            ),
        )
        conn.commit()


def _get_sync(db_path: str, artefact_id: str) -> dict | None:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM artefacts WHERE artefact_id = ? AND deleted_at IS NULL",
            (artefact_id,),
        ).fetchone()
    return dict(row) if row else None


def _soft_delete_sync(db_path: str, artefact_id: str) -> None:
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE artefacts SET deleted_at = ? WHERE artefact_id = ?",
            (now, artefact_id),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def init_artefact_db() -> None:
    await asyncio.to_thread(_init_sync, _DB_PATH)


async def save(
    session_id: str,
    content: str,
    filename: str,
    content_type: str = "text/markdown",
    ttl_days: int = _DEFAULT_TTL,
) -> dict:
    """Persist *content* and return ``{artefact_id, url}``."""
    artefact_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    if _BACKEND == "s3":
        storage_key = await _s3_upload(artefact_id, session_id, filename, content, content_type)
        url = await _s3_presign(storage_key)
    else:
        storage_key = await _local_write(artefact_id, filename, content)
        url = f"{_BASE_URL}/artefacts/{artefact_id}/download"

    await asyncio.to_thread(
        _insert_sync,
        _DB_PATH,
        {
            "artefact_id": artefact_id,
            "session_id": session_id,
            "filename": filename,
            "storage_key": storage_key,
            "content_type": content_type,
            "created_at": now,
            "ttl_days": ttl_days,
        },
    )
    return {"artefact_id": artefact_id, "url": url}


async def get(artefact_id: str) -> dict | None:
    return await asyncio.to_thread(_get_sync, _DB_PATH, artefact_id)


async def soft_delete(artefact_id: str) -> None:
    await asyncio.to_thread(_soft_delete_sync, _DB_PATH, artefact_id)


async def read_local(artefact_id: str) -> tuple[bytes, str] | None:
    """Return ``(file_bytes, content_type)`` for the local backend, or ``None`` if not found."""
    record = await get(artefact_id)
    if record is None:
        return None
    path = Path(record["storage_key"])
    if not path.exists():
        return None
    data = await asyncio.to_thread(path.read_bytes)
    return data, record["content_type"]


# ---------------------------------------------------------------------------
# Local backend
# ---------------------------------------------------------------------------


async def _local_write(artefact_id: str, filename: str, content: str) -> str:
    dir_ = _LOCAL_DIR / artefact_id
    await asyncio.to_thread(dir_.mkdir, parents=True, exist_ok=True)
    path = dir_ / filename
    await asyncio.to_thread(path.write_text, content, "utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# S3 backend (requires boto3 — install separately for production)
# ---------------------------------------------------------------------------


async def _s3_upload(
    artefact_id: str,
    session_id: str,
    filename: str,
    content: str,
    content_type: str,
) -> str:
    import boto3  # noqa: PLC0415

    s3_key = f"artefacts/{session_id}/{artefact_id}/{filename}"
    client = boto3.client("s3")
    await asyncio.to_thread(
        client.put_object,
        Bucket=_S3_BUCKET,
        Key=s3_key,
        Body=content.encode("utf-8"),
        ContentType=content_type,
    )
    return s3_key


async def _s3_presign(storage_key: str, expiry: int = 900) -> str:
    import boto3  # noqa: PLC0415

    client = boto3.client("s3")
    url = await asyncio.to_thread(
        client.generate_presigned_url,
        "get_object",
        Params={"Bucket": _S3_BUCKET, "Key": storage_key},
        ExpiresIn=expiry,
    )
    return url
