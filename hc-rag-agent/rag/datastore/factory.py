"""Production wiring: vector store singleton, paths, demo bootstrap.

Single entry point for “which index do we use?” — driven by env (``VECTOR_STORE_BACKEND``, paths).
Not used for unit tests that inject fake embeddings (patch :func:`get_embeddings` on this module).
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from core.config import VECTOR_STORE_BACKEND, VECTOR_STORE_DIR, VECTORDB_PATH
from rag.datastore.local import DictVectorIndex, DuckDBVectorIndex
from rag.datastore.opensearch import OpenSearchVectorIndex
from rag.embedding import get_embeddings

if TYPE_CHECKING:
    from rag.protocols import Retriever

log = structlog.get_logger(__name__)

_VECTOR_SINGLETON: Any | None = None


def _backend_name() -> str:
    raw = (
        (os.getenv("VECTOR_STORE_BACKEND") or VECTOR_STORE_BACKEND or "memory")
        .strip()
        .lower()
    )
    if raw not in ("memory", "duckdb", "opensearch"):
        log.warning("vectorstore.unknown_backend defaulting memory raw=%s", raw)
        return "memory"
    return raw


def _project_data_dir() -> Path:
    """Embeddings + ``document/`` bootstrap live under repo ``data/`` (not ``src/data``).

    Override with ``RAG_DATA_DIR`` for a custom tree.
    """
    env = os.getenv("RAG_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent.parent.parent / "data"


def _persist_base() -> Path:
    env = os.getenv("VECTOR_STORE_DIR")
    if env:
        return Path(env).expanduser()
    return Path(VECTOR_STORE_DIR)


def get_duckdb_index_path() -> Path:
    """Path to the DuckDB vector index. VECTORDB_PATH wins; falls back to VECTOR_STORE_DIR/clara.duckdb."""
    env = os.getenv("VECTORDB_PATH")
    if env:
        return Path(env).expanduser()
    return _persist_base() / "clara.duckdb"


def bootstrap_txt_corpus(
    index: DictVectorIndex | DuckDBVectorIndex | OpenSearchVectorIndex,
    input_directory: Path,
    output_json: Path | None,
) -> int:
    """One embedding per .txt file (legacy demo behaviour). Returns count added."""
    if not input_directory.is_dir():
        log.warning("vectorstore.bootstrap.skip_no_dir path=%s", input_directory)
        return 0

    from rag.schemas.chunks import Chunk, ChunkMetadata

    embedder = get_embeddings()
    json_records: list[dict[str, Any]] = []
    flat: list[tuple[str, str, dict[str, str], list[float]]] = []
    chunks: list[Chunk] = []

    for filename in sorted(input_directory.iterdir()):
        if not filename.name.endswith(".txt"):
            continue
        text = filename.read_text(encoding="utf-8")
        record_id = str(uuid.uuid4())
        vec = embedder.embed_documents([text])[0]
        json_records.append(
            {
                "id": record_id,
                "filename": filename.name,
                "content": text,
                "embedding": vec,
            },
        )
        flat.append(
            (
                record_id,
                text,
                {"filename": filename.name, "source": filename.name},
                vec,
            ),
        )
        chunks.append(
            Chunk(
                id=record_id,
                text=text,
                embedding=vec,
                metadata=ChunkMetadata(title=filename.name, doc_id=record_id),
            ),
        )

    if output_json is not None and json_records:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(json_records, indent=4, ensure_ascii=False), encoding="utf-8"
        )

    if isinstance(index, DictVectorIndex):
        index.upsert_flat(flat)
        return len(flat)

    if isinstance(index, DuckDBVectorIndex):
        index.upsert_blocking(chunks)
        return len(chunks)

    if isinstance(index, OpenSearchVectorIndex):
        index.upsert_blocking(chunks)
        return len(chunks)

    raise TypeError(type(index))


def _build_index() -> Any:
    backend = _backend_name()
    data_dir = _project_data_dir()
    doc_dir = data_dir / "document"

    if backend == "memory":
        dict_idx = DictVectorIndex()
        out_json = data_dir / "embeddings" / "all_text_embeddings.json"
        if dict_idx.load_from_json_file(out_json) > 0:
            log.info("vectorstore.memory.from_json path=%s", out_json)
        elif bootstrap_txt_corpus(dict_idx, doc_dir, out_json) > 0:
            log.info("vectorstore.memory.from_txt path=%s", doc_dir)
        return dict_idx

    if backend == "duckdb":
        try:
            import duckdb  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "VECTOR_STORE_BACKEND=duckdb requires the 'duckdb' package. "
                "Install with: uv sync",
            ) from e
        dbfile = get_duckdb_index_path()
        idx = DuckDBVectorIndex(dbfile)
        if idx.doc_count() == 0:
            added = bootstrap_txt_corpus(idx, doc_dir, None)
            log.info("vectorstore.duckdb.bootstrapped n=%s path=%s", added, dbfile)
        return idx

    if backend == "opensearch":
        try:
            import opensearchpy  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "VECTOR_STORE_BACKEND=opensearch requires the 'opensearch-py' package. "
                "Install optional deps: uv sync --extra rag",
            ) from e
        os_idx = OpenSearchVectorIndex.from_env()
        if os_idx.doc_count() == 0:
            added = bootstrap_txt_corpus(os_idx, doc_dir, None)
            log.info(
                "vectorstore.opensearch.bootstrapped n=%s index=%s",
                added,
                os_idx._index,
            )
        return os_idx

    raise RuntimeError(f"Unhandled VECTOR_STORE_BACKEND={backend!r}")


def get_vectorstore() -> Any:
    """Return the process-wide vector index (Retriever + ``similarity_search_with_score``)."""
    global _VECTOR_SINGLETON
    if _VECTOR_SINGLETON is None:
        _VECTOR_SINGLETON = _build_index()
    return _VECTOR_SINGLETON


def get_local_retriever() -> Retriever:
    """Same object as :func:`get_vectorstore` — use with :class:`~src.rag.preprocessing.ingestion.IngestionPipeline`."""
    return get_vectorstore()  # type: ignore[no-any-return]


def reset_vectorstore_for_tests() -> None:
    """Clear the singleton (used by tests only)."""
    global _VECTOR_SINGLETON
    _VECTOR_SINGLETON = None
