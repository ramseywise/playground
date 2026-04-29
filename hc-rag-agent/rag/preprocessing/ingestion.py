from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.preprocessing.base import Chunker
    from rag.retrieval.cache import RetrievalCache
    from rag.protocols import Embedder, Retriever
    from rag.retrieval.snippet import MetadataDB, SnippetDB

log = logging.getLogger(__name__)

# Sentence boundary: end with . ! ? followed by whitespace or end-of-string.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_MIN_SNIPPET_LEN = 30  # characters — discard very short fragments
_MAX_SNIPPET_LEN = 400  # characters — discard overly long sentences


@dataclass
class IngestionResult:
    """Summary of a single document ingestion run."""

    doc_id: str
    chunk_count: int
    snippet_count: int
    skipped: bool = field(default=False)  # True = checksum already present, doc skipped


class IngestionPipeline:
    """Orchestrates raw-text → vectorDB + metadataDB + snippetDB ingestion.

    Flow for each document:
      1. Compute SHA-256 checksum → skip if already ingested (idempotent)
      2. Chunk via *chunker* (Chunker protocol)
      3. Embed via *embedder* (Embedder protocol)
      4. Upsert chunks to *vector_store* (Retriever protocol)
      5. Extract short sentence snippets from raw text
      6. Write snippets to *snippet_db* (SnippetDB)
      7. Write doc metadata to *metadata_db* (MetadataDB)
    """

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: Retriever,
        metadata_db: MetadataDB,
        snippet_db: SnippetDB,
        retrieval_cache: RetrievalCache | None = None,
        batch_size: int = 64,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._metadata_db = metadata_db
        self._snippet_db = snippet_db
        self._retrieval_cache = retrieval_cache
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_document(self, doc: dict[str, str]) -> IngestionResult:
        """Ingest a single document dict.

        The dict must contain at minimum ``text``.  Optional keys:
        ``title``, ``url``, ``source``, ``content_type``, ``topic``, ``source_file``.
        """
        text = doc.get("text", "")
        if not text:
            log.warning(
                "ingestion.skip.empty source_file=%s",
                doc.get("source_file", ""),
            )
            return IngestionResult(
                doc_id="", chunk_count=0, snippet_count=0, skipped=True
            )

        checksum = _sha256(text)
        if self._metadata_db.document_exists_by_checksum(checksum):
            log.info(
                "ingestion.skip.duplicate source_file=%s",
                doc.get("source_file", ""),
            )
            return IngestionResult(
                doc_id="", chunk_count=0, snippet_count=0, skipped=True
            )

        stable = (doc.get("stable_doc_id") or "").strip()
        doc_id = stable or _stable_id(
            doc.get("source_file") or doc.get("title") or checksum
        )
        title = doc.get("title", "")
        source = doc.get("source", "")
        source_file = doc.get("source_file", "")
        content_type = doc.get("content_type", "")
        topic = doc.get("topic", "")

        # --- Chunking ---
        chunks = self._chunker.chunk_document(doc)
        log.info("ingestion.chunked doc_id=%s chunk_count=%d", doc_id, len(chunks))

        # --- Embedding ---
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_passages(texts)
        for chunk, emb in zip(chunks, embeddings, strict=True):
            chunk.embedding = emb

        # --- Vector store upsert ---
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]
            await self._vector_store.upsert(batch)

        # --- Snippet extraction ---
        sentences = self._extract_snippets(text)
        snippet_records = [
            {
                "id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "text": s,
                "title": title,
                "topic": topic,
                "position": idx,
                "source": source,
            }
            for idx, s in enumerate(sentences)
        ]
        self._snippet_db.insert_snippets(snippet_records)

        # --- Metadata write ---
        self._metadata_db.insert_document(
            doc_id,
            title=title,
            source=source,
            source_file=source_file,
            content_type=content_type,
            topic=topic,
            word_count=len(text.split()),
            chunk_count=len(chunks),
            snippet_count=len(sentences),
            checksum=checksum,
        )

        log.info(
            "ingestion.done doc_id=%s chunk_count=%d snippet_count=%d",
            doc_id,
            len(chunks),
            len(sentences),
        )
        if self._retrieval_cache is not None:
            self._retrieval_cache.clear()
        return IngestionResult(
            doc_id=doc_id,
            chunk_count=len(chunks),
            snippet_count=len(sentences),
        )

    async def ingest_documents(
        self, docs: list[dict[str, str]]
    ) -> list[IngestionResult]:
        """Ingest a list of document dicts, returning one result per doc."""
        results = []
        for doc in docs:
            result = await self.ingest_document(doc)
            results.append(result)
        return results

    async def ingest_file(self, path: Path) -> IngestionResult:
        """Load a Markdown file from *path* and ingest it."""
        from rag.preprocessing.loaders import load_markdown_file

        doc = load_markdown_file(path)
        return await self.ingest_document(doc)

    async def ingest_directory(
        self, directory: Path, glob_pattern: str = "*.md"
    ) -> list[IngestionResult]:
        """Load all matching files from *directory* and ingest them."""
        from rag.preprocessing.loaders import load_directory

        docs = load_directory(directory, glob_pattern)
        return await self.ingest_documents(docs)

    async def ingest_s3_object(
        self, bucket: str, key: str, region: str = ""
    ) -> IngestionResult:
        """Load a single object from S3 and ingest it."""
        import asyncio

        from rag.preprocessing.loaders import S3DocumentLoader

        loader = S3DocumentLoader(bucket=bucket, region=region)
        doc = await asyncio.to_thread(loader.load_object, key)
        return await self.ingest_document(doc)

    async def ingest_s3_prefix(
        self, bucket: str, prefix: str, region: str = ""
    ) -> list[IngestionResult]:
        """Load all matching objects under an S3 prefix and ingest them."""
        import asyncio

        from rag.preprocessing.loaders import S3DocumentLoader

        loader = S3DocumentLoader(bucket=bucket, region=region)
        docs = await asyncio.to_thread(loader.load_prefix, prefix)
        return await self.ingest_documents(docs)

    # ------------------------------------------------------------------
    # Snippet extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_snippets(
        text: str,
        min_len: int = _MIN_SNIPPET_LEN,
        max_len: int = _MAX_SNIPPET_LEN,
    ) -> list[str]:
        """Split *text* into sentences and return those within length bounds.

        Strips Markdown-style headings (lines starting with #) before splitting.
        """
        # Remove heading lines — they're structural noise for snippet search
        cleaned = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
        # Collapse excess whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        raw_sentences = _SENTENCE_RE.split(cleaned)
        snippets: list[str] = []
        for s in raw_sentences:
            s = s.strip()
            if min_len <= len(s) <= max_len:
                snippets.append(s)
        return snippets


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_id(seed: str) -> str:
    """Deterministic 16-char hex ID derived from *seed*."""
    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:16]  # noqa: S324
