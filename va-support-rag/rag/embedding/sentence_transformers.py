"""Embedding client — lazy init, no API key required.

Uses a local SentenceTransformer model. Configure via:

- ``EMBEDDING_MODEL`` — explicit Hugging Face / sentence-transformers model id (wins over profile).
- ``EMBEDDING_PROFILE`` — ``minilm`` | ``multilingual`` when ``EMBEDDING_MODEL`` is unset.
- ``EMBEDDING_MODEL_REVISION`` — optional HF revision (pin a commit).

E5-family models (e.g. ``intfloat/multilingual-e5-large``) get ``query:`` / ``passage:`` prefixes
automatically on encode paths.
"""

from __future__ import annotations

import contextlib
import io
import logging
import threading
from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings

from core.config import EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION

log = logging.getLogger(__name__)

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, Any] = {}


def _cache_key(model_name: str, revision: str | None) -> str:
    return f"{model_name}@{revision or 'default'}"


def load_sentence_transformer(model_name: str, revision: str | None = None) -> Any:
    """Load and cache a :class:`sentence_transformers.SentenceTransformer` process-wide."""
    key = _cache_key(model_name, revision)
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            for _name in (
                "sentence_transformers",
                "transformers",
                "huggingface_hub",
            ):
                logging.getLogger(_name).setLevel(logging.ERROR)

            log.info(
                "loading embedding model: %s revision=%s",
                model_name,
                revision or "latest",
            )
            kwargs: dict[str, Any] = {}
            if revision:
                kwargs["revision"] = revision
            with contextlib.redirect_stdout(io.StringIO()):
                _MODEL_CACHE[key] = SentenceTransformer(model_name, **kwargs)
            log.info("embedding model ready")
        return _MODEL_CACHE[key]


class _LocalEmbeddings(Embeddings):
    """LangChain-compatible wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        self._model_name = model_name
        self._revision = revision
        self._is_e5 = "e5" in model_name.lower()

    def _encode_model(self) -> Any:
        return load_sentence_transformer(self._model_name, self._revision)

    def embed_query(self, text: str) -> list[float]:
        prefix = "query: " if self._is_e5 else ""
        return (  # type: ignore[no-any-return]
            self._encode_model()
            .encode(prefix + text, normalize_embeddings=True, show_progress_bar=False)
            .tolist()
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefix = "passage: " if self._is_e5 else ""
        prefixed = [prefix + t for t in texts]
        return (  # type: ignore[no-any-return]
            self._encode_model()
            .encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
            .tolist()
        )


@lru_cache(maxsize=1)
def get_embeddings() -> _LocalEmbeddings:
    """Return the local embedding model (loaded on first use)."""
    return _LocalEmbeddings(EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION)


__all__ = [
    "get_embeddings",
    "load_sentence_transformer",
]
