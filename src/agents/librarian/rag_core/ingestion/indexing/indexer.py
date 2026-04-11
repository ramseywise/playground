"""Indexer: orchestrates the parse → chunk → embed → upsert pipeline.

ChunkIndexer is the single entry point for building a retrieval index from
raw documents. It wires together:
  - A Chunker (any strategy from preprocessing/chunker.py)
  - An Embedder (MultilingualEmbedder or MiniLMEmbedder from preprocessing/embedder.py)
  - A Retriever (ChromaRetriever, OpenSearchRetriever, or DuckDBRetriever)

Usage:
    indexer = ChunkIndexer(chunker=HtmlAwareChunker(), embedder=embedder, retriever=retriever)
    await indexer.index_documents(docs)      # async (Chroma/OpenSearch)
    await indexer.index_document(doc)        # single doc

All operations are async to match the Retriever.upsert() Protocol signature.
"""

from __future__ import annotations

from typing import Any

from agents.librarian.rag_core.ingestion.base import Chunker, ChunkerConfig
from agents.librarian.rag_core.ingestion.parsing.cleaning import clean_text
from agents.librarian.rag_core.ingestion.parsing.pipeline import preprocess_corpus
from agents.librarian.rag_core.retrieval.base import Embedder, Retriever
from agents.librarian.rag_core.schemas.chunks import Chunk
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)


class ChunkIndexer:
    """Orchestrates the ingest pipeline: raw doc → chunks → embeddings → vector store.

    Args:
        chunker:      Any object implementing the Chunker Protocol (chunk_document).
        embedder:     Any object implementing the Embedder Protocol (embed_passage/embed_passages).
        retriever:    Any object implementing the Retriever Protocol (upsert).
        batch_size:   Number of chunks to embed and upsert per batch. Controls memory pressure.
        clean_input:  If True, runs clean_text on each doc's text field before chunking.
    """

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        retriever: Retriever,
        batch_size: int = 64,
        clean_input: bool = True,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._retriever = retriever
        self._batch_size = batch_size
        self._clean_input = clean_input

    async def index_document(self, doc: dict) -> int:
        """Chunk, embed, and upsert a single document.

        Args:
            doc: Document dict with at least "url", "title", and "text" / "full_text" keys.

        Returns:
            Number of chunks indexed.
        """
        if self._clean_input:
            for field in ("text", "full_text", "content"):
                if doc.get(field):
                    doc = {**doc, field: clean_text(doc[field])}
                    break

        chunks = self._chunker.chunk_document(doc)
        if not chunks:
            log.warning("indexer.no_chunks", url=doc.get("url", ""))
            return 0

        indexed = await self._embed_and_upsert(chunks)
        log.info("indexer.document.done", url=doc.get("url", ""), n_chunks=indexed)
        return indexed

    async def index_documents(
        self,
        docs: list[dict],
        *,
        preprocess: bool = False,
        preprocess_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """Chunk, embed, and upsert a list of documents.

        Args:
            docs: List of document dicts.
            preprocess: If True, run preprocess_corpus() before chunking
                (cleaning, dedup, enrichment).
            preprocess_kwargs: Passed to preprocess_corpus() when preprocess=True.

        Returns:
            Dict with keys "documents" (processed) and "chunks" (total indexed).
        """
        if preprocess:
            docs = preprocess_corpus(docs, **(preprocess_kwargs or {}))

        total_chunks = 0
        for doc in docs:
            total_chunks += await self.index_document(doc)

        log.info("indexer.batch.done", n_docs=len(docs), n_chunks=total_chunks)
        return {"documents": len(docs), "chunks": total_chunks}

    async def _embed_and_upsert(self, chunks: list[Chunk]) -> int:
        """Embed chunks in batches and upsert into the retriever."""
        indexed = 0
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]
            texts = [c.text for c in batch]

            embeddings = self._embedder.embed_passages(texts)

            embedded_chunks = [
                chunk.model_copy(update={"embedding": vec})
                for chunk, vec in zip(batch, embeddings)
            ]

            await self._retriever.upsert(embedded_chunks)
            indexed += len(embedded_chunks)
            log.debug(
                "indexer.batch.upserted",
                batch_start=i,
                batch_size=len(embedded_chunks),
            )

        return indexed


# ---------------------------------------------------------------------------
# Source-type aware factory
# ---------------------------------------------------------------------------


def build_indexer_for_source(
    source_type: str,
    embedder: Embedder,
    retriever: Retriever,
    config: ChunkerConfig | None = None,
    batch_size: int = 64,
) -> ChunkIndexer:
    """Return a ChunkIndexer pre-configured for a named source type.

    Source types and their default chunkers:
        "html" / "docs"   → HtmlAwareChunker (heading-boundary recursive)
        "blog"            → HtmlAwareChunker (larger max_tokens for article bodies)
        "faq"             → FixedChunker(max_tokens=512, min_tokens=1) — single-chunk per Q&A pair
        "snippet"         → FixedChunker(max_tokens=512, min_tokens=1) — keep snippets whole
        "parent_doc"      → ParentDocChunker (two-level child/parent)
        "structured"      → StructuredChunker (recursive, no heading detection)
        "fixed"           → FixedChunker
        "overlapping"     → OverlappingChunker

    Raises:
        ValueError: If source_type is not recognised.
    """
    from agents.librarian.rag_core.ingestion.chunking.strategies import (
        AdjacencyChunker,
        FixedChunker,
        OverlappingChunker,
        StructuredChunker,
    )
    from agents.librarian.rag_core.ingestion.chunking.html_aware import HtmlAwareChunker
    from agents.librarian.rag_core.ingestion.chunking.parent_doc import ParentDocChunker

    cfg = config or ChunkerConfig()

    chunker: Chunker
    if source_type in ("html", "docs", "help"):
        chunker = HtmlAwareChunker(config=cfg)
    elif source_type == "blog":
        blog_cfg = ChunkerConfig(max_tokens=800, overlap_tokens=64, min_tokens=50)
        chunker = HtmlAwareChunker(config=blog_cfg)
    elif source_type in ("faq", "snippet"):
        faq_cfg = ChunkerConfig(max_tokens=512, overlap_tokens=0, min_tokens=1)
        chunker = FixedChunker(config=faq_cfg)
    elif source_type == "parent_doc":
        chunker = ParentDocChunker()
    elif source_type == "structured":
        chunker = StructuredChunker(config=cfg)
    elif source_type == "fixed":
        chunker = FixedChunker(config=cfg)
    elif source_type == "overlapping":
        chunker = OverlappingChunker(config=cfg)
    elif source_type == "adjacency":
        chunker = AdjacencyChunker(config=cfg)
    else:
        raise ValueError(f"Unknown source_type: {source_type!r}")

    return ChunkIndexer(
        chunker=chunker,
        embedder=embedder,
        retriever=retriever,
        batch_size=batch_size,
    )
