from __future__ import annotations

import os
from typing import Any

from librarian.schemas.chunks import Chunk
from librarian.schemas.retrieval import RetrievalResult
from librarian.config import settings
from core.logging import get_logger

log = get_logger(__name__)

# BM25 analyzer language — set OPENSEARCH_BM25_LANGUAGE to match your corpus.
# Wrong analyzer silently degrades recall on non-English text (e.g. using "english"
# stemmer on French will silently fail to stem French morphology).
_BM25_LANGUAGE = os.environ.get("OPENSEARCH_BM25_LANGUAGE", "english")


class OpenSearchRetriever:
    """Async OpenSearch retriever with hybrid BM25 + k-NN search.

    bm25_weight + vector_weight should sum to 1.0.
    BM25 analyzer is configured via OPENSEARCH_BM25_LANGUAGE env var (default: "english").
    """

    def __init__(
        self,
        index: str | None = None,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        verify_certs: bool = True,
    ) -> None:
        self.index = index or settings.opensearch_index
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.verify_certs = verify_certs
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from opensearchpy import AsyncOpenSearch  # type: ignore[import-untyped]

            self._client = AsyncOpenSearch(
                hosts=[settings.opensearch_url],
                http_auth=(settings.opensearch_user, settings.opensearch_password),
                use_ssl=settings.opensearch_url.startswith("https"),
                verify_certs=self.verify_certs,
            )
        return self._client

    async def upsert(self, chunks: list[Chunk]) -> None:
        client = self._get_client()
        actions: list[dict] = []
        for chunk in chunks:
            if chunk.embedding is None:
                log.warning("opensearch.upsert.missing_embedding", chunk_id=chunk.id)
                continue
            actions.append({"index": {"_index": self.index, "_id": chunk.id}})
            actions.append(
                {
                    "text": chunk.text,
                    "embedding": chunk.embedding,
                    "metadata": chunk.metadata.model_dump(),
                }
            )
        if actions:
            resp = await client.bulk(body=actions)
            if resp.get("errors"):
                log.error("opensearch.upsert.errors", index=self.index, n=len(chunks))
            else:
                log.info("opensearch.upsert.done", index=self.index, n=len(chunks))

    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        client = self._get_client()

        knn_query: dict = {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k,
                    "boost": self.vector_weight,
                }
            }
        }
        bm25_query: dict = {
            "match": {
                "text": {
                    "query": query_text,
                    "analyzer": _BM25_LANGUAGE,
                    "boost": self.bm25_weight,
                }
            }
        }

        query: dict = {"bool": {"should": [knn_query, bm25_query]}}
        if metadata_filter:
            query["bool"]["filter"] = [
                {"term": {f"metadata.{field}": val}}
                for field, val in metadata_filter.items()
            ]

        resp = await client.search(
            index=self.index,
            body={"query": query, "size": k},
        )

        results: list[RetrievalResult] = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            meta_data = src.get("metadata", {})
            from librarian.schemas.chunks import ChunkMetadata

            chunk = Chunk(
                id=hit["_id"],
                text=src["text"],
                embedding=src.get("embedding"),
                metadata=ChunkMetadata(**meta_data),
            )
            results.append(
                RetrievalResult(chunk=chunk, score=hit["_score"], source="hybrid")
            )
        return results

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
