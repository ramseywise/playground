"""Document parsing, cleaning, and deduplication for RAG preprocessing.

Pipeline entry point: preprocess_corpus() runs the full sequence:
    clean_text → detect_language (optional filter) → deduplicate_exact → enrich_documents

Individual functions are also importable for use in custom pipelines.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


_NOISE_PATTERNS: list[str] = [
    r"Sent from my \w+",
    r"Get Outlook for .*",
    r"--\s*\n.*$",  # email signature separator
]


def clean_text(text: str, extra_noise_patterns: list[str] | None = None) -> str:
    """Normalize whitespace and strip common noise (email signatures, app footers).

    Args:
        text: Raw input text.
        extra_noise_patterns: Additional regex patterns to strip (e.g. brand-specific boilerplate).

    Returns:
        Cleaned text with normalized whitespace.
    """
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    for pattern in _NOISE_PATTERNS + (extra_noise_patterns or []):
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    return text.strip()


def remove_boilerplate(text: str, patterns: list[str] | None = None) -> str:
    """Remove boilerplate content from documents.

    Args:
        text: Input text.
        patterns: Regex patterns to strip. Applied in addition to nothing (no defaults —
            pass domain-specific patterns explicitly to keep this function generic).

    Returns:
        Text with matched patterns removed.
    """
    if not text or not patterns:
        return text or ""

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    return text.strip()


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def detect_language(text: str) -> dict[str, Any]:
    """Heuristic language detection — does not require an external library.

    Returns a dict with keys:
        language: "unknown" | "uncertain" | detected language code(s)
        confidence: float [0, 1]
        issues: list of detected non-target script names

    Note: This is a lightweight heuristic for filtering, not a replacement for
    langdetect/fasttext. For production multi-language corpora use a proper detector.
    """
    if not text or len(text.strip()) < 10:
        return {"language": "unknown", "confidence": 0.0, "issues": ["too_short"]}

    issues: list[str] = []

    if re.search(r"[\u0400-\u04FF]", text):
        issues.append("cyrillic")
    if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text):
        issues.append("cjk")
    if re.search(r"[\u0600-\u06FF]", text):
        issues.append("arabic")

    if issues:
        return {"language": "non_latin", "confidence": 0.9, "issues": issues}

    return {"language": "latin", "confidence": 0.8, "issues": []}


def filter_by_language(
    documents: list[dict],
    allowed_languages: set[str] | None = None,
    text_field: str = "text",
) -> list[dict]:
    """Filter documents by detected language.

    Args:
        documents: List of document dicts.
        allowed_languages: Set of allowed language codes (e.g. {"latin", "uncertain"}).
            Defaults to {"latin", "unknown"} — keeps anything without a detected script issue.
        text_field: Field name containing text content.

    Returns:
        Filtered list of documents.
    """
    if allowed_languages is None:
        allowed_languages = {"latin", "unknown"}

    kept: list[dict] = []
    filtered = 0

    for doc in documents:
        text = doc.get(text_field) or doc.get("content", "")
        lang = detect_language(text)
        if lang["language"] in allowed_languages:
            kept.append(doc)
        else:
            filtered += 1

    log.info("parsing.filter_language.done", kept=len(kept), filtered=filtered)
    return kept


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def compute_text_hash(text: str, normalize: bool = True) -> str:
    """MD5 hash of text for exact-duplicate detection.

    Args:
        text: Input text.
        normalize: If True, lowercases and collapses whitespace before hashing.

    Returns:
        Hex MD5 digest.
    """
    if normalize:
        text = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(text.encode()).hexdigest()


def deduplicate_exact(
    documents: list[dict],
    text_field: str = "text",
    keep: str = "first",
) -> list[dict]:
    """Remove exact-duplicate documents by text hash.

    Args:
        documents: List of document dicts.
        text_field: Field containing text to hash.
        keep: "first" keeps earliest occurrence; "last" keeps latest.

    Returns:
        Deduplicated list preserving original order.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    duplicates = 0

    items = documents if keep == "first" else list(reversed(documents))

    for doc in items:
        text = doc.get(text_field) or doc.get("content", "")
        h = compute_text_hash(text)
        if h not in seen:
            seen.add(h)
            unique.append(doc)
        else:
            duplicates += 1

    result = unique if keep == "first" else list(reversed(unique))
    log.info("parsing.dedup_exact.done", kept=len(result), removed=duplicates)
    return result


def deduplicate_fuzzy(
    documents: list[dict],
    embedder: Any,
    text_field: str = "text",
    threshold: float = 0.95,
) -> list[dict]:
    """Remove near-duplicate documents using embedding cosine similarity.

    Requires scikit-learn. Embeds all documents and removes any pair with
    cosine similarity >= threshold, keeping the earlier document.

    Args:
        documents: List of document dicts.
        embedder: Object with an embed_passages(texts) method returning list[list[float]].
        text_field: Field containing text to embed.
        threshold: Cosine similarity threshold for near-duplicates (default 0.95).

    Returns:
        Deduplicated list.
    """
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

    texts = [doc.get(text_field) or doc.get("content", "") for doc in documents]
    embeddings = embedder.embed_passages([t[:500] for t in texts])

    sim_matrix = cosine_similarity(embeddings)
    to_remove: set[int] = set()

    for i in range(len(sim_matrix)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(sim_matrix)):
            if sim_matrix[i, j] >= threshold:
                to_remove.add(j)

    result = [doc for i, doc in enumerate(documents) if i not in to_remove]
    log.info("parsing.dedup_fuzzy.done", kept=len(result), removed=len(to_remove))
    return result


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------


def extract_metadata(text: str, url: str | None = None) -> dict[str, Any]:
    """Extract lightweight metadata from document text and URL.

    Returns a dict with: word_count, char_count, and optionally source / type
    fields inferred from URL structure and content heuristics.

    Args:
        text: Document text.
        url: Optional source URL used to infer source category.

    Returns:
        Metadata dict (never raises).
    """
    meta: dict[str, Any] = {
        "word_count": len(text.split()),
        "char_count": len(text),
    }

    if url:
        if "/blog" in url:
            meta["source"] = "blog"
        elif "/api" in url or "/reference" in url:
            meta["source"] = "api_docs"
        elif "/help" in url or "/support" in url or "/docs" in url:
            meta["source"] = "help_center"

        id_match = re.search(r"/articles?/(\d+)", url)
        if id_match:
            meta["article_id"] = id_match.group(1)

    # Content-type heuristics
    if re.search(r"\bstep\s+\d+\b|\bhow\s+to\b", text, re.IGNORECASE):
        meta["content_type"] = "procedural"
    elif re.search(r"\bfaq\b|^\s*q\s*:", text, re.IGNORECASE | re.MULTILINE):
        meta["content_type"] = "faq"

    return meta


def enrich_documents(documents: list[dict], text_field: str = "text") -> list[dict]:
    """Enrich documents with metadata extracted from text and URL.

    Merges extract_metadata() results into each doc dict (does not overwrite
    existing keys).

    Args:
        documents: List of document dicts.
        text_field: Field containing text.

    Returns:
        New list of dicts with metadata fields added.
    """
    enriched: list[dict] = []
    for doc in documents:
        text = doc.get(text_field) or doc.get("content", "")
        url = doc.get("url", "")
        meta = extract_metadata(text, url)
        enriched.append({**meta, **doc})  # doc wins on key conflicts

    log.info("parsing.enrich.done", n=len(enriched))
    return enriched


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_corpus(
    documents: list[dict],
    text_field: str = "text",
    *,
    clean: bool = True,
    filter_language: bool = False,
    allowed_languages: set[str] | None = None,
    deduplicate: bool = True,
    enrich: bool = True,
    boilerplate_patterns: list[str] | None = None,
    noise_patterns: list[str] | None = None,
) -> list[dict]:
    """Run full preprocessing pipeline on a document corpus.

    Stages (each independently togglable):
        1. clean_text + remove_boilerplate (if clean=True)
        2. filter_by_language (if filter_language=True)
        3. deduplicate_exact (if deduplicate=True)
        4. enrich_documents (if enrich=True)

    Args:
        documents: Raw document dicts, each with at least a ``text_field`` key.
        text_field: Name of the text field in each document.
        clean: Run text cleaning and boilerplate removal.
        filter_language: Filter documents whose detected language is not in allowed_languages.
        allowed_languages: Passed to filter_by_language (default: {"latin", "unknown"}).
        deduplicate: Remove exact-duplicate documents.
        enrich: Add word_count, source, content_type metadata.
        boilerplate_patterns: Domain-specific regex patterns for remove_boilerplate.
        noise_patterns: Additional patterns for clean_text.

    Returns:
        Preprocessed document list.
    """
    log.info("parsing.preprocess.start", n_docs=len(documents))
    result = list(documents)

    if clean:
        for doc in result:
            text = doc.get(text_field) or doc.get("content", "")
            text = clean_text(text, extra_noise_patterns=noise_patterns)
            if boilerplate_patterns:
                text = remove_boilerplate(text, boilerplate_patterns)
            doc[text_field] = text

    if filter_language:
        result = filter_by_language(result, allowed_languages, text_field)

    if deduplicate:
        result = deduplicate_exact(result, text_field)

    if enrich:
        result = enrich_documents(result, text_field)

    log.info("parsing.preprocess.done", n_in=len(documents), n_out=len(result))
    return result
