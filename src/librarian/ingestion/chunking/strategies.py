"""All chunking strategies for the librarian RAG pipeline.

Strategies (in order of complexity):
  FixedChunker          — word-window splits, no overlap. Baseline.
  OverlappingChunker    — fixed size + overlap_tokens context carry.
  StructuredChunker     — recursive paragraph → sentence → word split, no heading pass.
  HtmlAwareChunker      — heading-boundary first, recursive fallback. Primary strategy.
  AdjacencyChunker      — fixed splits with neighbour lookup via chunk_id index.
  ParentDocChunker      — two-level: small children indexed, large parents for generation.

All strategies implement the Chunker Protocol from base.py:
    def chunk_document(self, doc: dict) -> list[Chunk]

Shared split helpers (_recursive_split, _merge_with_overlap, _hard_split_text,
_fixed_split, _overlapping_split) are module-level to allow direct import from
HtmlAwareChunker and ParentDocChunker without circular deps.
"""

from __future__ import annotations

import re

from librarian.ingestion.base import ChunkerConfig
from librarian.ingestion.chunking.utils import (
    WORDS_PER_TOKEN,
    approx_tokens,
    hard_split_text,
    make_chunk,
    make_doc_id,
    merge_with_overlap,
    recursive_split_with_config,
)
from librarian.schemas.chunks import Chunk, ChunkMetadata

# ---------------------------------------------------------------------------
# Backward-compat aliases (underscore-prefixed names)
# ---------------------------------------------------------------------------
_WORDS_PER_TOKEN = WORDS_PER_TOKEN
_approx_tokens = approx_tokens
_make_doc_id = make_doc_id
_make_chunk = make_chunk
_hard_split_text = hard_split_text
_merge_with_overlap = merge_with_overlap
_recursive_split = recursive_split_with_config


def _fixed_split(
    text: str, url: str, title: str, section: str, doc_id: str, config: ChunkerConfig
) -> list[Chunk]:
    """Split by word count with no overlap."""
    words = text.split()
    step = max(1, int(config.max_tokens * _WORDS_PER_TOKEN))

    chunks: list[Chunk] = []
    for i in range(0, len(words), step):
        chunk_text = " ".join(words[i : i + step])
        if _approx_tokens(chunk_text) >= config.min_tokens:
            chunk_section = (
                f"{section}#{len(chunks)}"
                if chunks or i + step < len(words)
                else section
            )
            chunks.append(
                _make_chunk(
                    chunk_text,
                    url,
                    title,
                    chunk_section,
                    _make_doc_id(url, chunk_section),
                )
            )

    return chunks if chunks else [_make_chunk(text, url, title, section, doc_id)]


def _overlapping_split(
    text: str, url: str, title: str, section: str, doc_id: str, config: ChunkerConfig
) -> list[Chunk]:
    """Split by word count with overlap_tokens overlap between adjacent chunks."""
    words = text.split()
    step = max(1, int(config.max_tokens * _WORDS_PER_TOKEN))
    overlap_w = int(config.overlap_tokens * _WORDS_PER_TOKEN)

    chunks: list[Chunk] = []
    i = 0
    while i < len(words):
        chunk_text = " ".join(words[i : i + step])
        if _approx_tokens(chunk_text) >= config.min_tokens:
            chunk_section = f"{section}#{len(chunks)}"
            chunks.append(
                _make_chunk(
                    chunk_text,
                    url,
                    title,
                    chunk_section,
                    _make_doc_id(url, chunk_section),
                )
            )
        i += max(1, step - overlap_w)

    return chunks if chunks else [_make_chunk(text, url, title, section, doc_id)]


# ---------------------------------------------------------------------------
# Heading split helper (shared by HtmlAwareChunker and ParentDocChunker)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,3}\s+.+|[A-Z][A-Z0-9 :/-]{2,60})$", re.MULTILINE)


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split plain text into (heading, body) pairs at h1–h3 and ALL-CAPS headings."""
    positions = [m.start() for m in _HEADING_RE.finditer(text)]
    if not positions:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    if positions[0] > 0:
        sections.append(("", text[: positions[0]]))

    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        line_end = text.index("\n", pos) if "\n" in text[pos:end] else end
        heading = text[pos:line_end].strip().lstrip("#").strip()
        body = text[line_end:end].strip()
        sections.append((heading, body))

    return sections


# ---------------------------------------------------------------------------
# Strategy: FixedChunker
# ---------------------------------------------------------------------------


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
        doc_id = _make_doc_id(url, section)
        return _fixed_split(text, url, title, section, doc_id, self.config)


# ---------------------------------------------------------------------------
# Strategy: OverlappingChunker
# ---------------------------------------------------------------------------


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
        doc_id = _make_doc_id(url, section)
        return _overlapping_split(text, url, title, section, doc_id, self.config)


# ---------------------------------------------------------------------------
# Strategy: StructuredChunker
# ---------------------------------------------------------------------------


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
        doc_id = _make_doc_id(url, section)

        if _approx_tokens(text) <= self.config.max_tokens:
            if _approx_tokens(text) >= self.config.min_tokens:
                return [_make_chunk(text, url, title, section, doc_id)]
            return []

        sub_texts = _recursive_split(text, self.config)
        chunks: list[Chunk] = []
        for i, sub_text in enumerate(sub_texts):
            chunk_section = f"{section}#{i}" if len(sub_texts) > 1 else section
            chunks.append(
                _make_chunk(
                    sub_text,
                    url,
                    title,
                    chunk_section,
                    _make_doc_id(url, chunk_section),
                )
            )
        return chunks


# ---------------------------------------------------------------------------
# Strategy: AdjacencyChunker
# ---------------------------------------------------------------------------


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

        doc_id = _make_doc_id(url, section)
        words = text.split()
        step = max(1, int(self.config.max_tokens * _WORDS_PER_TOKEN))

        chunks: list[Chunk] = []
        i = 0
        while i < len(words):
            chunk_text = " ".join(words[i : i + step])
            if _approx_tokens(chunk_text) >= self.config.min_tokens:
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
                            doc_id=_make_doc_id(url, chunk_section),
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
