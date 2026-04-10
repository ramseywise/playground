"""Full preprocessing pipeline orchestrator."""

from __future__ import annotations

from agents.librarian.ingestion.parsing.cleaning import (
    clean_text,
    remove_boilerplate,
)
from agents.librarian.ingestion.parsing.dedup import deduplicate_exact
from agents.librarian.ingestion.parsing.enrichment import enrich_documents
from agents.librarian.ingestion.parsing.language import filter_by_language
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)


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
