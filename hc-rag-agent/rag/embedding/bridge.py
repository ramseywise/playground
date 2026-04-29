"""Bridge LangChain :class:`~langchain_core.embeddings.Embeddings` to :class:`~app.rag.retrieval.protocols.Embedder`."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


class LangChainEmbeddingsBridge:
    """Wraps a sync LangChain embedder with async helpers for EnsembleRetriever."""

    def __init__(self, embeddings: Embeddings) -> None:
        self._emb = embeddings

    def embed_query(self, text: str) -> list[float]:
        return list(self._emb.embed_query(text))

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    def embed_passage(self, text: str) -> list[float]:
        out = self._emb.embed_documents([text])
        return list(out[0])

    async def aembed_passage(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_passage, text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [list(v) for v in self._emb.embed_documents(texts)]

    async def aembed_passages(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_passages, texts)


__all__ = ["LangChainEmbeddingsBridge"]
