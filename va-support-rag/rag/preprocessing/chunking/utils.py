"""Shared chunking utilities — token/word estimates and split primitives."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

from rag.preprocessing.base import ChunkerConfig
from rag.schemas.chunks import Chunk, ChunkMetadata

WORDS_PER_TOKEN: float = 0.75  # conservative estimate for multilingual text


def approx_tokens(text: str) -> int:
    """Approximate token count using word-count / WORDS_PER_TOKEN."""
    return max(1, int(len(text.split()) / WORDS_PER_TOKEN))


def word_count(text: str) -> int:
    """Raw word count (used by the separator-based splitter)."""
    return len(text.split())


def make_doc_id(url: str, section: str | None) -> str:
    """Deterministic document ID from URL + section title.

    Uses ``{url}::{section or ''}`` as the hash input.
    """
    raw = f"{url}::{section or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def stable_doc_id_from_document(doc: dict) -> str:
    """Human-readable stable id per document (legacy notebook / DuckDB style).

    Priority:
      1. ``doc["stable_doc_id"]`` when set
      2. Help-center article URLs: ``.../articles/12345-...`` → ``help_12345``
      3. Blog: ``blog_{slug}`` from last URL path segment
      4. Fallback: :func:`make_doc_id` with URL + section
    """
    explicit = (doc.get("stable_doc_id") or "").strip()
    if explicit:
        return explicit
    url = (doc.get("url") or "").strip()
    source = (doc.get("source") or "").strip().lower()
    m = re.search(r"/articles?/(\d+)", url)
    if m:
        return f"help_{m.group(1)}"
    path_parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    if path_parts and ("blog" in url.lower() or source == "blog"):
        slug = re.sub(r"[^\w]+", "_", path_parts[-1])[:80].strip("_") or "post"
        return f"blog_{slug}"
    if "faq" in url.lower() or source == "faq":
        return f"faq_{make_doc_id(url, '')}"
    return make_doc_id(url, doc.get("section") or "")


def make_chunk(
    text: str,
    url: str,
    title: str,
    section: str,
    doc_id: str,
    *,
    chunk_index: int | None = None,
    chunk_id_mode: str = "hash",
) -> Chunk:
    """Create a Chunk: hash ids (default) or stable ``{doc_id}_{chunk_index}``."""
    if chunk_id_mode == "stable" and chunk_index is not None:
        chunk_id = f"{doc_id}_{chunk_index}"
    else:
        chunk_id = hashlib.sha256(f"{doc_id}:{text[:64]}".encode()).hexdigest()[:20]
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(url=url, title=title, section=section, doc_id=doc_id),
    )


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

    return chunks


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
    Used by StructuredChunker (via ``recursive_split_with_config``).
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


def recursive_split_by_separators(
    text: str, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """Split text using progressively finer separators until chunks fit max_tokens.

    Tries separators in order: paragraph -> line -> sentence -> word.
    Uses raw word_count for size estimation.
    Used by HtmlAwareChunker and ParentDocChunker in ``strategies``.
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
                    # NB: [-0:] slices the *whole* list — use [] when overlap_tokens == 0
                    overlap_words = (
                        current.split()[-overlap_tokens:]
                        if current and overlap_tokens > 0
                        else []
                    )
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
