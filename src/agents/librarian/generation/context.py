"""Post-retrieval context optimization utilities.

Provides source-diversity enforcement and XML context formatting for LLM
prompts.  Extracted from legacy ``orchestration/layers/context_optimization.py``
and adapted for the current ``RankedChunk`` schema.
"""

from __future__ import annotations

from agents.librarian.schemas.chunks import RankedChunk


def deduplicate_by_source(
    chunks: list[RankedChunk],
    *,
    max_per_source: int = 2,
) -> list[RankedChunk]:
    """Limit chunks per source URL for diversity.

    Assumes *chunks* are already sorted by relevance (rank order).
    Keeps up to ``max_per_source`` items from any single URL.

    Args:
        chunks: Ranked chunks in relevance order.
        max_per_source: Maximum items from the same source URL.

    Returns:
        Diverse subset preserving original order.

    """
    source_counts: dict[str, int] = {}
    diverse: list[RankedChunk] = []

    for rc in chunks:
        source = rc.chunk.metadata.url
        count = source_counts.get(source, 0)
        if count < max_per_source:
            diverse.append(rc)
            source_counts[source] = count + 1

    return diverse


def format_as_xml_context(chunks: list[RankedChunk]) -> str:
    """Format ranked chunks as XML for LLM context injection.

    Wraps each chunk in a ``<document>`` element with metadata attributes,
    making it easy for the LLM to cite sources and assess provenance.

    Args:
        chunks: Ranked chunks to format.

    Returns:
        XML-formatted context string.

    """
    if not chunks:
        return ""

    parts: list[str] = []

    for i, rc in enumerate(chunks):
        meta = rc.chunk.metadata
        updated_tag = (
            f"\n<updated>{meta.last_updated}</updated>" if meta.last_updated else ""
        )
        xml = (
            f'<document index="{i + 1}" source="{meta.url}" '
            f'relevance="{rc.relevance_score:.2f}">\n'
            f"<title>{meta.title}</title>\n"
            f"<url>{meta.url}</url>"
            f"{updated_tag}\n"
            f"<content>\n{rc.chunk.text}\n</content>\n"
            f"</document>"
        )
        parts.append(xml)

    return "\n\n".join(parts)
