"""Shared chunking utilities used across all chunking strategies.

Consolidates helpers from strategies.py and html_aware.py to eliminate
duplication while preserving both splitting algorithms unchanged.
"""

from __future__ import annotations

import hashlib
import re

from agents.librarian.rag_core.ingestion.base import ChunkerConfig
from agents.librarian.rag_core.schemas.chunks import Chunk, ChunkMetadata

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORDS_PER_TOKEN: float = 0.75  # conservative estimate for multilingual text

# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def approx_tokens(text: str) -> int:
    """Approximate token count using word-count / WORDS_PER_TOKEN."""
    return max(1, int(len(text.split()) / WORDS_PER_TOKEN))


def word_count(text: str) -> int:
    """Raw word count (used by the separator-based splitter)."""
    return len(text.split())


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def make_doc_id(url: str, section: str | None) -> str:
    """Deterministic document ID from URL + section title.

    Uses ``{url}::{section or ''}`` as the hash input.
    """
    raw = f"{url}::{section or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def make_chunk(text: str, url: str, title: str, section: str, doc_id: str) -> Chunk:
    """Create a Chunk with a deterministic ID derived from doc_id + text prefix."""
    chunk_id = hashlib.sha256(f"{doc_id}:{text[:64]}".encode()).hexdigest()[:20]
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(url=url, title=title, section=section, doc_id=doc_id),
    )


# ---------------------------------------------------------------------------
# Splitting primitives (config-based — used by strategies.py)
# ---------------------------------------------------------------------------


def hard_split_text(text: str, config: ChunkerConfig) -> list[str]:
    """Word-window split with overlap — raw strings (no Chunk objects)."""
    words = text.split()
    step = max(1, int(config.max_tokens * WORDS_PER_TOKEN))
    overlap_w = int(config.overlap_tokens * WORDS_PER_TOKEN)
    min_t = config.min_tokens

    chunks: list[str] = []
    i = 0
    while i < len(words):
        window = words[i : i + step]
        chunk_text = " ".join(window)
        if approx_tokens(chunk_text) >= min_t:
            chunks.append(chunk_text)
        i += max(1, step - overlap_w)

    return chunks if chunks else [text]


def merge_with_overlap(pieces: list[str], config: ChunkerConfig) -> list[str]:
    """Greedily merge pieces up to max_tokens, carrying overlap_tokens of context."""
    chunks: list[str] = []
    current_words: list[str] = []

    for piece in pieces:
        piece_words = piece.split()
        if not piece_words:
            continue
        tentative = current_words + piece_words
        if approx_tokens(" ".join(tentative)) <= config.max_tokens:
            current_words = tentative
        else:
            if current_words:
                text_out = " ".join(current_words)
                if approx_tokens(text_out) >= config.min_tokens:
                    chunks.append(text_out)
                current_words = current_words[-config.overlap_tokens :] + piece_words
            else:
                chunks.extend(hard_split_text(" ".join(piece_words), config))
                current_words = []

    if current_words:
        text_out = " ".join(current_words)
        if approx_tokens(text_out) >= config.min_tokens:
            chunks.append(text_out)

    return chunks if chunks else [" ".join(pieces)]


def recursive_split_with_config(text: str, config: ChunkerConfig) -> list[str]:
    """Split: paragraph boundaries -> sentence boundaries -> word-window.

    Uses approx_tokens (word_count / 0.75) for size estimation.
    Used by StructuredChunker.
    """
    if approx_tokens(text) <= config.max_tokens:
        return [text] if approx_tokens(text) >= config.min_tokens else []

    paragraphs = re.split(r"\n{2,}", text)
    if len(paragraphs) > 1:
        return merge_with_overlap(paragraphs, config)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        return merge_with_overlap(sentences, config)

    return hard_split_text(text, config)


# ---------------------------------------------------------------------------
# Splitting primitives (raw-param — used by html_aware.py, parent_doc.py)
# ---------------------------------------------------------------------------


def recursive_split_by_separators(
    text: str, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """Split text using progressively finer separators until chunks fit max_tokens.

    Tries separators in order: paragraph -> line -> sentence -> word.
    Uses raw word_count for size estimation.
    Used by HtmlAwareChunker, ParentDocChunker.
    """
    if word_count(text) <= max_tokens:
        return [text]

    for sep in ("\n\n", "\n", ". ", " "):
        parts = text.split(sep)
        if len(parts) > 1:
            chunks: list[str] = []
            current = ""
            for part in parts:
                candidate = (current + sep + part).strip() if current else part.strip()
                if word_count(candidate) <= max_tokens:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # carry overlap from tail of previous chunk
                    overlap_words = current.split()[-overlap_tokens:] if current else []
                    current = (" ".join(overlap_words) + " " + part).strip()
            if current:
                chunks.append(current)
            # recurse only if we made progress
            if len(chunks) > 1:
                result: list[str] = []
                for c in chunks:
                    result.extend(
                        recursive_split_by_separators(c, max_tokens, overlap_tokens)
                    )
                return result

    # fallback: word-level hard split
    words = text.split()
    return [
        " ".join(words[i : i + max_tokens])
        for i in range(0, len(words), max_tokens - overlap_tokens)
        if words[i : i + max_tokens]
    ]
