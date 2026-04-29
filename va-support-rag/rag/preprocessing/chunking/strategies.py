"""Chunking strategies for the RAG preprocessing pipeline.

- FixedChunker / OverlappingChunker — word-window splits (baseline + overlap).
- StructuredChunker — recursive paragraph → sentence → word split, no heading pass.
- AdjacencyChunker — fixed splits with neighbour lookup via chunk_id index.
- HtmlAwareChunker — heading-boundary first, recursive fallback.
- ParentDocChunker — two-level: small children indexed, large parents for generation.

All strategies implement the Chunker Protocol from base.py:
    def chunk_document(self, doc: dict) -> list[Chunk]
"""

from __future__ import annotations

import re
import uuid

from rag.preprocessing.base import ChunkerConfig
from rag.preprocessing.chunking.utils import (
    WORDS_PER_TOKEN,
    approx_tokens,
    make_chunk,
    make_doc_id,
    recursive_split_by_separators,
    recursive_split_with_config,
    stable_doc_id_from_document,
    word_count,
)
from rag.schemas.chunks import Chunk, ChunkMetadata


def _fixed_split(
    text: str,
    url: str,
    title: str,
    section: str,
    doc_id: str,
    config: ChunkerConfig,
    *,
    canonical_doc_id: str | None = None,
) -> list[Chunk]:
    """Split by word count with no overlap."""
    words = text.split()
    step = max(1, int(config.max_tokens * WORDS_PER_TOKEN))
    stable = config.chunk_id_mode == "stable" and canonical_doc_id

    chunks: list[Chunk] = []
    for i in range(0, len(words), step):
        chunk_text = " ".join(words[i : i + step])
        if approx_tokens(chunk_text) >= config.min_tokens:
            chunk_section = (
                f"{section}#{len(chunks)}"
                if chunks or i + step < len(words)
                else section
            )
            if stable:
                idx = len(chunks)
                chunks.append(
                    make_chunk(
                        chunk_text,
                        url,
                        title,
                        chunk_section,
                        canonical_doc_id,  # type: ignore[arg-type]
                        chunk_index=idx,
                        chunk_id_mode="stable",
                    )
                )
            else:
                chunks.append(
                    make_chunk(
                        chunk_text,
                        url,
                        title,
                        chunk_section,
                        make_doc_id(url, chunk_section),
                    )
                )

    if chunks:
        return chunks
    if stable:
        return [
            make_chunk(
                text,
                url,
                title,
                section,
                canonical_doc_id,  # type: ignore[arg-type]
                chunk_index=0,
                chunk_id_mode="stable",
            )
        ]
    return [make_chunk(text, url, title, section, doc_id)]


def _overlapping_split(
    text: str,
    url: str,
    title: str,
    section: str,
    doc_id: str,
    config: ChunkerConfig,
    *,
    canonical_doc_id: str | None = None,
) -> list[Chunk]:
    """Split by word count with overlap_tokens overlap between adjacent chunks."""
    words = text.split()
    step = max(1, int(config.max_tokens * WORDS_PER_TOKEN))
    overlap_w = int(config.overlap_tokens * WORDS_PER_TOKEN)
    stable = config.chunk_id_mode == "stable" and canonical_doc_id

    chunks: list[Chunk] = []
    i = 0
    while i < len(words):
        chunk_text = " ".join(words[i : i + step])
        if approx_tokens(chunk_text) >= config.min_tokens:
            chunk_section = f"{section}#{len(chunks)}"
            if stable:
                idx = len(chunks)
                chunks.append(
                    make_chunk(
                        chunk_text,
                        url,
                        title,
                        chunk_section,
                        canonical_doc_id,  # type: ignore[arg-type]
                        chunk_index=idx,
                        chunk_id_mode="stable",
                    )
                )
            else:
                chunks.append(
                    make_chunk(
                        chunk_text,
                        url,
                        title,
                        chunk_section,
                        make_doc_id(url, chunk_section),
                    )
                )
        i += max(1, step - overlap_w)

    if chunks:
        return chunks
    if stable:
        return [
            make_chunk(
                text,
                url,
                title,
                section,
                canonical_doc_id,  # type: ignore[arg-type]
                chunk_index=0,
                chunk_id_mode="stable",
            )
        ]
    return [make_chunk(text, url, title, section, doc_id)]


class FixedChunker:
    """Hard word-count splits, no overlap. Baseline benchmark.

    Use when: evaluating chunking impact, or for structured data where
    context continuity doesn't matter (e.g. FAQ answer bodies).
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk_document(self, doc: dict) -> list[Chunk]:
        url = doc.get("url", "")
        title = doc.get("title", "")
        section = doc.get("section", "")
        text = (doc.get("text") or doc.get("full_text") or "").strip()
        if not text:
            return []
        doc_id = make_doc_id(url, section)
        canonical = (
            stable_doc_id_from_document(doc)
            if self.config.chunk_id_mode == "stable"
            else None
        )
        return _fixed_split(
            text,
            url,
            title,
            section,
            doc_id,
            self.config,
            canonical_doc_id=canonical,
        )


class OverlappingChunker:
    """Fixed size + overlap_tokens carried between chunks.

    Better recall than FixedChunker for queries that span chunk boundaries.
    Trade-off: ~(overlap/max_tokens) more chunks → larger index.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk_document(self, doc: dict) -> list[Chunk]:
        url = doc.get("url", "")
        title = doc.get("title", "")
        section = doc.get("section", "")
        text = (doc.get("text") or doc.get("full_text") or "").strip()
        if not text:
            return []
        doc_id = make_doc_id(url, section)
        canonical = (
            stable_doc_id_from_document(doc)
            if self.config.chunk_id_mode == "stable"
            else None
        )
        return _overlapping_split(
            text,
            url,
            title,
            section,
            doc_id,
            self.config,
            canonical_doc_id=canonical,
        )


class StructuredChunker:
    """Recursive paragraph → sentence → word split. No heading pre-pass.

    Same fallback logic as HtmlAwareChunker but applied to the full text
    without section detection. Good for prose without markdown headings.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk_document(self, doc: dict) -> list[Chunk]:
        url = doc.get("url", "")
        title = doc.get("title", "")
        section = doc.get("section", "")
        text = (doc.get("text") or doc.get("full_text") or "").strip()
        if not text:
            return []
        doc_id = make_doc_id(url, section)
        stable = self.config.chunk_id_mode == "stable"
        canonical = stable_doc_id_from_document(doc) if stable else None

        if approx_tokens(text) <= self.config.max_tokens:
            if approx_tokens(text) >= self.config.min_tokens:
                if stable:
                    return [
                        make_chunk(
                            text,
                            url,
                            title,
                            section,
                            canonical,  # type: ignore[arg-type]
                            chunk_index=0,
                            chunk_id_mode="stable",
                        )
                    ]
                return [make_chunk(text, url, title, section, doc_id)]
            return []

        sub_texts = recursive_split_with_config(text, self.config)
        chunks: list[Chunk] = []
        for i, sub_text in enumerate(sub_texts):
            chunk_section = f"{section}#{i}" if len(sub_texts) > 1 else section
            if stable:
                chunks.append(
                    make_chunk(
                        sub_text,
                        url,
                        title,
                        chunk_section,
                        canonical,  # type: ignore[arg-type]
                        chunk_index=i,
                        chunk_id_mode="stable",
                    )
                )
            else:
                chunks.append(
                    make_chunk(
                        sub_text,
                        url,
                        title,
                        chunk_section,
                        make_doc_id(url, chunk_section),
                    )
                )
        return chunks


class AdjacencyChunker:
    """Fixed splits with positional chunk IDs enabling neighbour lookup at query time.

    chunk_id format: ``{doc_id}_chunk{i}``

    After chunk_document(), call neighbors(chunk_id) to get
    (prev_chunk_id, next_chunk_id) for context-window expansion at retrieval time.

    Note: see VectorGraph RAG systems for an embedding-similarity graph approach
    instead of linear adjacency.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()
        self._last_doc_id: str = ""
        self._last_count: int = 0

    def chunk_document(self, doc: dict) -> list[Chunk]:
        url = doc.get("url", "")
        title = doc.get("title", "")
        section = doc.get("section", "")
        text = (doc.get("text") or doc.get("full_text") or "").strip()

        if not text:
            self._last_doc_id = ""
            self._last_count = 0
            return []

        doc_id = make_doc_id(url, section)
        words = text.split()
        step = max(1, int(self.config.max_tokens * WORDS_PER_TOKEN))

        chunks: list[Chunk] = []
        i = 0
        while i < len(words):
            chunk_text = " ".join(words[i : i + step])
            if approx_tokens(chunk_text) >= self.config.min_tokens:
                idx = len(chunks)
                chunk_section = f"{section}_chunk{idx}"
                chunks.append(
                    Chunk(
                        id=f"{doc_id}_chunk{idx}",
                        text=chunk_text,
                        metadata=ChunkMetadata(
                            url=url,
                            title=title,
                            section=chunk_section,
                            doc_id=make_doc_id(url, chunk_section),
                        ),
                    )
                )
            i += step

        self._last_doc_id = doc_id
        self._last_count = len(chunks)
        return chunks

    def neighbors(self, chunk_id: str) -> tuple[str | None, str | None]:
        """Return (prev_chunk_id, next_chunk_id) for chunk_id.

        Raises RuntimeError if chunk_document() has not been called.
        Raises ValueError if chunk_id format is invalid.
        """
        if self._last_count == 0:
            raise RuntimeError("Call chunk_document() before calling neighbors().")
        try:
            base_id, idx_str = chunk_id.rsplit("_chunk", 1)
            idx = int(idx_str)
        except ValueError as exc:
            raise ValueError(f"Bad chunk_id format: {chunk_id!r}") from exc

        prev_id = f"{base_id}_chunk{idx - 1}" if idx > 0 else None
        next_id = f"{base_id}_chunk{idx + 1}" if (idx + 1) < self._last_count else None
        return prev_id, next_id


# Matches heading markers left by html-to-text converters, e.g. "## Heading"
_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


class HtmlAwareChunker:
    """Splits plain-text documents at heading boundaries, then recursively by size.

    Handles both ``doc["text"]`` and ``doc["full_text"]`` field names to accommodate
    variance across source connectors.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk_document(self, doc: dict) -> list[Chunk]:
        text: str = doc.get("text") or doc.get("full_text") or ""
        url: str = doc.get("url", "")
        title: str = doc.get("title", "")
        language: str = doc.get("language", "en")
        extra_meta: dict = {
            k: v
            for k, v in doc.items()
            if k not in {"text", "full_text", "url", "title", "language"}
        }

        sections = self._split_sections(text)
        chunks: list[Chunk] = []

        for section_title, body in sections:
            for fragment in recursive_split_by_separators(
                body.strip(),
                self.config.max_tokens,
                self.config.overlap_tokens,
            ):
                if word_count(fragment) < self.config.min_tokens:
                    continue
                doc_id = make_doc_id(url, section_title)
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        text=fragment,
                        metadata=ChunkMetadata(
                            url=url,
                            title=title,
                            doc_id=doc_id,
                            section=section_title,
                            language=language,
                            namespace=extra_meta.get("namespace"),
                            topic=extra_meta.get("topic"),
                            content_type=extra_meta.get("content_type"),
                            access_tier=extra_meta.get("access_tier"),
                            source_id=extra_meta.get("source_id"),
                        ),
                    )
                )

        return chunks

    def _split_sections(self, text: str) -> list[tuple[str | None, str]]:
        """Return list of (section_title, body) pairs split at h1–h3 headings."""
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [(None, text)]

        sections: list[tuple[str | None, str]] = []
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

        for i, match in enumerate(matches):
            section_title = match.group(1).strip()
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            sections.append((section_title, body))

        return sections


class ParentDocChunker:
    """Two-level chunking strategy.

    Parent chunks (full sections) are stored for generation context.
    Child chunks (small, overlapping) are indexed for retrieval and tagged with ``parent_id``.

    Child chunk IDs follow the pattern ``{parent_id}_child{i}``.
    """

    def __init__(
        self,
        parent_config: ChunkerConfig | None = None,
        child_config: ChunkerConfig | None = None,
    ) -> None:
        self.parent_config = parent_config or ChunkerConfig(
            max_tokens=512, overlap_tokens=0, min_tokens=1
        )
        self.child_config = child_config or ChunkerConfig(
            max_tokens=128, overlap_tokens=32, min_tokens=20
        )
        self._section_chunker = HtmlAwareChunker(config=self.parent_config)

    def chunk_document(self, doc: dict) -> list[Chunk]:
        """Return child chunks tagged with parent_id. Parents are embedded inline as metadata."""
        parent_chunks = self._section_chunker.chunk_document(doc)
        all_chunks: list[Chunk] = []

        for parent in parent_chunks:
            parent_id = str(uuid.uuid4())
            children = recursive_split_by_separators(
                parent.text,
                self.child_config.max_tokens,
                self.child_config.overlap_tokens,
            )
            for i, child_text in enumerate(children):
                if word_count(child_text) < self.child_config.min_tokens:
                    continue
                child_meta = ChunkMetadata(
                    url=parent.metadata.url,
                    title=parent.metadata.title,
                    doc_id=parent.metadata.doc_id,
                    section=parent.metadata.section,
                    language=parent.metadata.language,
                    parent_id=parent_id,
                    namespace=parent.metadata.namespace,
                    topic=parent.metadata.topic,
                    content_type=parent.metadata.content_type,
                    access_tier=parent.metadata.access_tier,
                    source_id=parent.metadata.source_id,
                )
                all_chunks.append(
                    Chunk(
                        id=f"{parent_id}_child{i}",
                        text=child_text,
                        metadata=child_meta,
                    )
                )

        return all_chunks
