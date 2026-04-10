from __future__ import annotations

import asyncio

from agents.librarian.utils.config import settings
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)

_MODEL_CACHE: dict[str, object] = {}


def _load_model(model_name: str) -> object:
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        log.info("embedder.load", model=model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class MultilingualEmbedder:
    """SentenceTransformer wrapper with E5 prefix enforcement.

    embed_query  → prepends "query: "   (search-time)
    embed_passage → prepends "passage: " (index-time)

    Model is loaded once at first use and cached process-wide.
    Configure via EMBEDDING_MODEL env var (default: intfloat/multilingual-e5-large).
    Swap to intfloat/e5-large-v2 for English-only corpora (~20% faster, same dim).
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.embedding_model

    @property
    def _model(self) -> object:
        return _load_model(self._model_name)

    def embed_query(self, text: str) -> list[float]:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        model: SentenceTransformer = self._model  # type: ignore[assignment]
        return model.encode(f"query: {text}", normalize_embeddings=True).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    def embed_passage(self, text: str) -> list[float]:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        model: SentenceTransformer = self._model  # type: ignore[assignment]
        return model.encode(f"passage: {text}", normalize_embeddings=True).tolist()

    async def aembed_passage(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_passage, text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        model: SentenceTransformer = self._model  # type: ignore[assignment]
        prefixed = [f"passage: {t}" for t in texts]
        return model.encode(prefixed, normalize_embeddings=True).tolist()

    async def aembed_passages(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_passages, texts)


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 (384-dim) — no prefix required.

    Fast, lightweight English-only embedder. Good for local dev, CI, or corpora
    where multilingual support is not needed.

    Trade-off vs MultilingualEmbedder:
      - 3× faster inference, ~6× smaller (22M vs 560M params)
      - English-only; degrades silently on other languages
      - 384-dim vs 1024-dim — lower resolution for tight nearest-neighbour search
    """

    _DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self._DEFAULT_MODEL

    @property
    def _model(self) -> object:
        return _load_model(self._model_name)

    def embed_query(self, text: str) -> list[float]:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        model: SentenceTransformer = self._model  # type: ignore[assignment]
        return model.encode(text, normalize_embeddings=True).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    def embed_passage(self, text: str) -> list[float]:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        model: SentenceTransformer = self._model  # type: ignore[assignment]
        return model.encode(text, normalize_embeddings=True).tolist()

    async def aembed_passage(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_passage, text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        model: SentenceTransformer = self._model  # type: ignore[assignment]
        return model.encode(texts, normalize_embeddings=True).tolist()

    async def aembed_passages(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_passages, texts)
