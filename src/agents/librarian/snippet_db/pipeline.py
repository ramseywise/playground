"""Snippet DB pipeline: extract → ground → store.

TODO(2): Implement the full pipeline once QAPairGenerator and SnippetStore
have concrete implementations. Current scaffold defines the orchestration
contract and data flow.
"""

from __future__ import annotations

from agents.librarian.snippet_db.base import QAPairGenerator, SnippetStore
from agents.librarian.snippet_db.models import QAPair, Snippet
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)


async def run_snippet_pipeline(
    documents: list[dict],
    generator: QAPairGenerator,
    store: SnippetStore,
    *,
    snippet_type: str = "text",
    text_field: str = "text",
) -> dict[str, int]:
    """Extract snippets from documents, generate QA pairs, and store both.

    Args:
        documents: Raw document dicts with at least a text field.
        generator: QAPairGenerator implementation (e.g. LLM-based).
        store: SnippetStore implementation (e.g. InMemory, DuckDB).
        snippet_type: Default snippet type tag.
        text_field: Field name containing text content.

    Returns:
        Dict with counts: {"snippets": N, "qa_pairs": M}.
    """
    # Step 1: Extract snippets from documents
    snippets = _extract_snippets(
        documents, snippet_type=snippet_type, text_field=text_field
    )
    if not snippets:
        log.warning("snippet_pipeline.no_snippets")
        return {"snippets": 0, "qa_pairs": 0}

    log.info("snippet_pipeline.extracted", n_snippets=len(snippets))

    # Step 2: Store snippets
    await store.upsert_snippets(snippets)

    # Step 3: Generate grounded QA pairs
    qa_pairs = await generator.generate(snippets)
    log.info("snippet_pipeline.generated", n_qa_pairs=len(qa_pairs))

    # Step 4: Store QA pairs
    await store.upsert_qa_pairs(qa_pairs)

    return {"snippets": len(snippets), "qa_pairs": len(qa_pairs)}


def _extract_snippets(
    documents: list[dict],
    *,
    snippet_type: str = "text",
    text_field: str = "text",
) -> list[Snippet]:
    """Extract self-contained snippets from documents.

    TODO(2): Add code-block extraction, config-block detection, and
    CLI command identification. Currently treats each document as a
    single snippet.
    """
    import hashlib

    snippets: list[Snippet] = []
    for doc in documents:
        text = doc.get(text_field) or doc.get("content", "")
        if not text.strip():
            continue

        snippet_id = hashlib.sha256(text[:256].encode()).hexdigest()[:16]
        snippets.append(
            Snippet(
                id=snippet_id,
                text=text,
                source_url=doc.get("url", ""),
                source_title=doc.get("title", ""),
                language=doc.get("language", "en"),
                snippet_type=snippet_type,
                tags=doc.get("tags", []),
            )
        )

    return snippets
