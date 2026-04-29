"""Document parsing — cleaning, dedup, language detection, enrichment.

Submodules:
- corpus.py:      preprocess_corpus (orchestrates stages below)
- cleaning.py:    clean_text, remove_boilerplate
- dedup.py:       compute_text_hash, deduplicate_exact, deduplicate_fuzzy
- language.py:    detect_language, filter_by_language
- enrichment.py:  extract_metadata, enrich_documents
"""

from rag.preprocessing.parsing.cleaning import (
    clean_text,
    remove_boilerplate,
)
from rag.preprocessing.parsing.dedup import (
    compute_text_hash,
    deduplicate_exact,
    deduplicate_fuzzy,
)
from rag.preprocessing.parsing.enrichment import (
    enrich_documents,
    extract_metadata,
)
from rag.preprocessing.parsing.language import (
    detect_language,
    filter_by_language,
)
from rag.preprocessing.parsing.corpus import preprocess_corpus

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
