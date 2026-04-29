"""OpenSearch kNN vector index — implements :class:`~app.rag.retrieval.protocols.Retriever`.

Requires optional dependency ``opensearch-py`` (``pip install 'help-support-rag-agent[rag]'``).

Index mapping uses ``knn_vector`` (Lucene HNSW) with inner product on normalized embeddings
(same convention as :class:`~app.rag.datastore.DuckDBVectorIndex`).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

from langchain_core.documents import Document

from rag.datastore.local import _metadata_to_chunk
from rag.embedding import get_embeddings
from rag.schemas.chunks import Chunk
from rag.schemas.retrieval import RetrievalResult

log = logging.getLogger(__name__)


def _parse_opensearch_hosts(hosts_csv: str) -> tuple[list[dict[str, Any]], bool]:
    """Return OpenSearch ``hosts`` list and whether TLS should be used."""
    out: list[dict[str, Any]] = []
    use_tls = False
    for part in hosts_csv.split(","):
        p = part.strip()
        if not p:
            continue
        url = p if "://" in p else f"http://{p}"
        u = urlparse(url)
        host = u.hostname or "localhost"
        port = u.port or (443 if u.scheme == "https" else 9200)
        if u.scheme == "https":
            use_tls = True
        out.append({"host": host, "port": port})
    if not out:
        out = [{"host": "127.0.0.1", "port": 9200}]
    return out, use_tls


class OpenSearchVectorIndex:
    """kNN-backed chunk store (cosine-style ranking via inner product on unit vectors)."""

    def __init__(
        self,
        *,
        hosts: list[dict[str, Any]],
        index_name: str,
        http_auth: tuple[str, str] | None = None,
        use_ssl: bool = False,
        verify_certs: bool = True,
        timeout: int = 30,
    ) -> None:
        from opensearchpy import OpenSearch  # type: ignore[import-untyped]

        self._client = OpenSearch(
            hosts=hosts,
            http_auth=http_auth,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_show_warn=False,
            timeout=timeout,
        )
        self._index = index_name
        self._embeddings = get_embeddings()
        self._dim: int | None = None

    @classmethod
    def from_env(cls) -> OpenSearchVectorIndex:
        from core.config import (
            OPENSEARCH_HOSTS,
            OPENSEARCH_INDEX,
            OPENSEARCH_PASSWORD,
            OPENSEARCH_TIMEOUT,
            OPENSEARCH_USE_SSL,
            OPENSEARCH_USER,
            OPENSEARCH_VERIFY_CERTS,
        )

        hosts, tls_from_url = _parse_opensearch_hosts(OPENSEARCH_HOSTS)
        use_ssl = OPENSEARCH_USE_SSL or tls_from_url
        auth: tuple[str, str] | None = None
        if OPENSEARCH_USER and OPENSEARCH_PASSWORD:
            auth = (OPENSEARCH_USER, OPENSEARCH_PASSWORD)
        return cls(
            hosts=hosts,
            index_name=OPENSEARCH_INDEX,
            http_auth=auth,
            use_ssl=use_ssl,
            verify_certs=OPENSEARCH_VERIFY_CERTS,
            timeout=OPENSEARCH_TIMEOUT,
        )

    def _ensure_index(self, embedding_dim: int) -> None:
        if self._dim == embedding_dim and self._client.indices.exists(
            index=self._index
        ):
            return
        if self._client.indices.exists(index=self._index):
            mapping = self._client.indices.get_mapping(index=self._index)
            props = (
                mapping.get(self._index, {}).get("mappings", {}).get("properties", {})
            )
            emb = props.get("embedding", {})
            if emb.get("dimension") and int(emb["dimension"]) != embedding_dim:
                raise ValueError(
                    f"OpenSearch index {self._index!r} dimension {emb.get('dimension')} "
                    f"!= model output {embedding_dim}",
                )
            self._dim = embedding_dim
            return

        body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "innerproduct",
                            "engine": "lucene",
                            "parameters": {"m": 16, "ef_construction": 100},
                        },
                    },
                    "text": {"type": "text"},
                    "url": {"type": "keyword"},
                    "title": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "topic": {"type": "keyword"},
                    "section": {"type": "keyword"},
                }
            },
        }
        self._client.indices.create(index=self._index, body=body)
        self._dim = embedding_dim
        log.info(
            "opensearch.index.created name=%s dim=%d",
            self._index,
            embedding_dim,
        )

    def doc_count(self) -> int:
        try:
            r = self._client.count(index=self._index)
            return int(r.get("count", 0))
        except Exception:
            return 0

    def _source_from_chunk(self, c: Chunk) -> dict[str, Any]:
        m = c.metadata
        return {
            "text": c.text,
            "url": m.url or "",
            "title": m.title or "",
            "doc_id": m.doc_id or "",
            "source": m.source_id or "",
            "topic": m.topic or "",
            "section": m.section or "",
            "embedding": c.embedding,
        }

    def upsert_blocking(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        from opensearchpy.helpers import bulk  # type: ignore[import-untyped]

        first = chunks[0]
        if first.embedding is None:
            raise ValueError(f"Chunk {first.id!r} has no embedding")
        dim = len(first.embedding)
        self._ensure_index(dim)

        actions: list[dict[str, Any]] = []
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.id!r} has no embedding")
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self._index,
                    "_id": c.id,
                    "_source": self._source_from_chunk(c),
                },
            )
        success, failed = bulk(self._client, actions, refresh="wait_for")
        if failed:
            log.warning(
                "opensearch.bulk.partial_failure success=%s failed_sample=%s",
                success,
                failed[:1],
            )
        log.debug("opensearch.bulk success=%s", success)

    async def upsert(self, chunks: list[Chunk]) -> None:
        await asyncio.to_thread(self.upsert_blocking, chunks)

    def _knn_search_sync(
        self,
        query_vector: list[float],
        k: int,
    ) -> list[RetrievalResult]:
        body = {
            "size": k,
            "query": {
                "knn": {"embedding": {"vector": query_vector, "k": k}},
            },
        }
        res = self._client.search(index=self._index, body=body)
        hits = res.get("hits", {}).get("hits", [])
        out: list[RetrievalResult] = []
        for hit in hits:
            src = hit.get("_source") or {}
            eid = str(hit.get("_id", ""))
            text = str(src.get("text", ""))
            score = float(hit.get("_score", 0.0))
            meta = {
                "url": src.get("url") or "",
                "title": src.get("title") or "",
                "doc_id": src.get("doc_id") or "",
                "id": eid,
                "source": src.get("source") or "",
                "topic": src.get("topic") or "",
                "section": src.get("section") or "",
            }
            ch = _metadata_to_chunk(meta, text, eid)
            out.append(
                RetrievalResult(
                    chunk=ch,
                    score=score,
                    source=str(src.get("url") or ""),
                ),
            )
        return out

    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        _ = query_text
        _ = metadata_filter
        if self.doc_count() == 0:
            return []
        return await asyncio.to_thread(self._knn_search_sync, query_vector, k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        if self.doc_count() == 0:
            return []
        qv = self._embeddings.embed_query(query)
        raw = self._knn_search_sync(qv, k)
        out: list[tuple[Document, float]] = []
        for r in raw:
            m = r.chunk.metadata
            doc = Document(
                page_content=r.chunk.text,
                metadata={
                    "id": r.chunk.id,
                    "url": m.url or "",
                    "title": m.title or "",
                    "doc_id": m.doc_id or "",
                    "source": m.source_id or "",
                    "topic": m.topic or "",
                    "section": m.section or "",
                },
            )
            out.append((doc, r.score))
        return out


__all__ = ["OpenSearchVectorIndex"]
