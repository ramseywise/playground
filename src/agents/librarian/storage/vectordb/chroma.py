from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.librarian.retrieval.scoring import term_overlap
from agents.librarian.schemas.chunks import Chunk, ChunkMetadata
from agents.librarian.schemas.retrieval import RetrievalResult
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_PERSIST_DIR = ".chroma"
_DEFAULT_COLLECTION = "librarian-chunks"

# BM25-like term overlap used as sparse signal alongside Chroma's cosine distance.
# Chroma only returns vector scores; we blend manually to mirror InMemoryRetriever.
_BM25_WEIGHT = 0.3
_VECTOR_WEIGHT = 0.7


def _chroma_distance_to_score(distance: float) -> float:
    """Convert Chroma cosine distance [0, 2] → similarity score [0, 1].

    Chroma returns L2 or cosine *distance* (lower = more similar).
    With cosine space: distance = 1 - cosine_similarity, so similarity = 1 - distance.
    Clamped to [0, 1] to handle floating-point edge cases.
    """
    return max(0.0, min(1.0, 1.0 - distance))


class ChromaRetriever:
    """Persistent ChromaDB retriever with hybrid vector + term-overlap scoring.

    Requires ``chromadb`` package (``uv add chromadb``).
    Data persists to ``persist_dir`` on disk — no Docker required.

    Hybrid score = bm25_weight * term_overlap + vector_weight * cosine_similarity
    which mirrors InMemoryRetriever's interface for test/prod parity.
    """

    def __init__(
        self,
        persist_dir: str | Path = _DEFAULT_PERSIST_DIR,
        collection_name: str = _DEFAULT_COLLECTION,
        bm25_weight: float = _BM25_WEIGHT,
        vector_weight: float = _VECTOR_WEIGHT,
    ) -> None:
        self._persist_dir = str(persist_dir)
        self._collection_name = collection_name
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self._client: Any | None = None
        self._collection: Any | None = None

    def _get_collection(self) -> Any:
        if self._collection is None:
            import chromadb  # type: ignore[import-untyped]

            if self._client is None:
                log.info("chroma.client.init", persist_dir=self._persist_dir)
                self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def upsert(self, chunks: list[Chunk]) -> None:
        collection = self._get_collection()
        if not chunks:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            if chunk.embedding is None:
                log.warning("chroma.upsert.missing_embedding", chunk_id=chunk.id)
                continue
            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            meta = chunk.metadata.model_dump()
            # Chroma metadata values must be str | int | float | bool — strip None
            metadatas.append({k: v for k, v in meta.items() if v is not None})

        if ids:
            collection.upsert(
                ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
            )
            log.info("chroma.upsert.done", n=len(ids), collection=self._collection_name)

    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        collection = self._get_collection()

        where: dict | None = None
        if metadata_filter:
            # Chroma $and syntax for multiple filters
            conditions = [{key: {"$eq": val}} for key, val in metadata_filter.items()]
            where = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        resp = collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, max(1, collection.count())),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results: list[RetrievalResult] = []
        ids = resp["ids"][0]
        documents = resp["documents"][0]
        metadatas_list = resp["metadatas"][0]
        distances = resp["distances"][0]

        for chunk_id, text, meta, dist in zip(
            ids, documents, metadatas_list, distances
        ):
            vec_score = _chroma_distance_to_score(dist)
            kw_score = term_overlap(query_text, text)
            hybrid_score = self.bm25_weight * kw_score + self.vector_weight * vec_score

            chunk = Chunk(
                id=chunk_id,
                text=text,
                metadata=ChunkMetadata(**meta),
            )
            results.append(
                RetrievalResult(chunk=chunk, score=hybrid_score, source="hybrid")
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
