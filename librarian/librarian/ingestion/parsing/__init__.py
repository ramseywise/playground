"""Document parsing — cleaning, dedup, language detection, enrichment.

Submodules:
- cleaning.py:    clean_text, remove_boilerplate
- dedup.py:       compute_text_hash, deduplicate_exact, deduplicate_fuzzy
- language.py:    detect_language, filter_by_language
- enrichment.py:  extract_metadata, enrich_documents
- pipeline.py:    preprocess_corpus (full pipeline orchestrator)
"""

from librarian.ingestion.parsing.cleaning import (
    clean_text,
    remove_boilerplate,
)
from librarian.ingestion.parsing.dedup import (
    compute_text_hash,
    deduplicate_exact,
    deduplicate_fuzzy,
)
from librarian.ingestion.parsing.enrichment import (
    enrich_documents,
    extract_metadata,
)
from librarian.ingestion.parsing.language import (
    detect_language,
    filter_by_language,
)
from librarian.ingestion.parsing.pipeline import preprocess_corpus

__all__ = [
    "clean_text",
    "compute_text_hash",
    "deduplicate_exact",
    "deduplicate_fuzzy",
    "detect_language",
    "enrich_documents",
    "extract_metadata",
    "filter_by_language",
    "preprocess_corpus",
    "remove_boilerplate",
]
