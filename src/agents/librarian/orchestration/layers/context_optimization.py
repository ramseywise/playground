"""Context optimization layer between reranker and generator.

Components:
- Deduplication (remove redundant chunks)
- Ordering optimization (recent first, logical flow)
- Context augmentation (add metadata as XML tags)
- Diversity enforcement (different sources in top-k)
"""

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rag_system.src.rag_core.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# CONTEXT ITEM
# =============================================================================


@dataclass
class ContextItem:
    """Single context item with metadata."""

    text: str
    url: str
    title: str
    source: str
    similarity: float
    date: str | None = None
    doc_type: str | None = None
    version: str | None = None

    @classmethod
    def from_retrieval_result(cls, doc: dict, score: float) -> "ContextItem":
        """Create from retrieval result."""
        return cls(
            text=doc.get("Text") or doc.get("text") or doc.get("content", ""),
            url=doc.get("URL") or doc.get("url", ""),
            title=doc.get("Title") or doc.get("title", ""),
            source=doc.get("source") or doc.get("Source", "unknown"),
            similarity=score,
            date=doc.get("date") or doc.get("updated_at"),
            doc_type=doc.get("type") or doc.get("doc_type"),
            version=doc.get("version") or doc.get("Version"),
        )


# =============================================================================
# DEDUPLICATION
# =============================================================================


def compute_text_fingerprint(text: str) -> str:
    """Compute fingerprint for deduplication."""
    # Normalize text
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    # Use first 500 chars for fingerprint
    return hashlib.md5(normalized[:500].encode()).hexdigest()


def deduplicate_semantic(
    items: list[ContextItem],
    embedder: Callable | None = None,
    threshold: float = 0.85,
) -> list[ContextItem]:
    """Remove semantically redundant chunks.

    Args:
        items: Context items to deduplicate
        embedder: Optional embedding function
        threshold: Similarity threshold for duplicates

    Returns:
        Deduplicated items

    """
    if len(items) <= 1:
        return items

    if embedder is None:
        # Fall back to fingerprint-based deduplication
        return deduplicate_fingerprint(items)

    from sklearn.metrics.pairwise import cosine_similarity

    # Embed all texts
    texts = [item.text[:500] for item in items]
    embeddings = embedder(texts)

    # Find duplicates
    to_remove = set()
    for i in range(len(embeddings)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(embeddings)):
            if j in to_remove:
                continue
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if sim >= threshold:
                # Remove the one with lower retrieval score
                if items[i].similarity >= items[j].similarity:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break

    unique = [item for i, item in enumerate(items) if i not in to_remove]
    logger.debug(f"Deduplicated {len(items)} → {len(unique)} items")
    return unique


def deduplicate_fingerprint(items: list[ContextItem]) -> list[ContextItem]:
    """Deduplicate using text fingerprints."""
    seen = set()
    unique = []

    for item in items:
        fp = compute_text_fingerprint(item.text)
        if fp not in seen:
            seen.add(fp)
            unique.append(item)

    return unique


def deduplicate_by_source(
    items: list[ContextItem], max_per_source: int = 2
) -> list[ContextItem]:
    """Limit chunks per source for diversity.

    Args:
        items: Context items (assumed sorted by relevance)
        max_per_source: Maximum items from same source

    Returns:
        Diverse items

    """
    source_counts = {}
    diverse = []

    for item in items:
        source = item.source or item.url
        count = source_counts.get(source, 0)

        if count < max_per_source:
            diverse.append(item)
            source_counts[source] = count + 1

    return diverse


# =============================================================================
# ORDERING OPTIMIZATION
# =============================================================================


def order_by_recency(items: list[ContextItem]) -> list[ContextItem]:
    """Order items with recent content first.

    Useful for tax/legal content where recency matters.
    """

    def parse_date(date_str: str | None) -> datetime:
        if not date_str:
            return datetime.min
        try:
            # Try common formats
            for fmt in ["%Y-%m-%d", "%d.%m.%Y", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    return datetime.strptime(date_str[:10], fmt)
                except ValueError:
                    continue
        except Exception:
            pass
        return datetime.min

    return sorted(items, key=lambda x: parse_date(x.date), reverse=True)


def order_by_type_priority(
    items: list[ContextItem],
    priority: list[str] = None,
) -> list[ContextItem]:
    """Order items by document type priority.

    Args:
        items: Context items
        priority: Type priority list (first = highest)

    Returns:
        Reordered items

    """
    if priority is None:
        priority = ["faq", "help_center", "tax_law", "tutorial", "blog", "community"]

    def get_priority(item: ContextItem) -> int:
        doc_type = (item.doc_type or item.source or "").lower()
        for i, p in enumerate(priority):
            if p in doc_type:
                return i
        return len(priority)

    return sorted(items, key=get_priority)


def order_procedural(items: list[ContextItem]) -> list[ContextItem]:
    """Order for procedural/step-by-step queries.

    Tries to maintain logical flow based on step indicators.
    """

    def get_step_number(item: ContextItem) -> int:
        # Look for "Schritt X" or step indicators
        match = re.search(r"Schritt\s*(\d+)|Step\s*(\d+)|(\d+)\.", item.text[:100])
        if match:
            return int(match.group(1) or match.group(2) or match.group(3))
        return 999

    return sorted(items, key=get_step_number)


# =============================================================================
# CONTEXT AUGMENTATION
# =============================================================================


def augment_with_metadata(items: list[ContextItem]) -> list[dict]:
    """Add metadata as XML tags for LLM context.

    Args:
        items: Context items

    Returns:
        Augmented documents for generator

    """
    augmented = []

    for i, item in enumerate(items):
        # Build metadata header
        meta_parts = []
        if item.title:
            meta_parts.append(f"Title: {item.title}")
        if item.source:
            meta_parts.append(f"Source: {item.source}")
        if item.date:
            meta_parts.append(f"Updated: {item.date}")
        if item.version:
            meta_parts.append(f"Version: {item.version}")

        metadata = " | ".join(meta_parts) if meta_parts else ""

        augmented_doc = {
            "Text": item.text,
            "URL": item.url,
            "Title": item.title or f"Document {i + 1}",
            "Similarity": round(item.similarity, 3),
            "Version": item.version or "N/A",
            "_metadata": metadata,
            "_source_type": item.source,
        }
        augmented.append(augmented_doc)

    return augmented


def format_as_xml_context(items: list[ContextItem]) -> str:
    """Format context items as XML for LLM.

    Args:
        items: Context items

    Returns:
        XML-formatted context string

    """
    parts = []

    for i, item in enumerate(items):
        xml = f"""<document index="{i + 1}" source="{item.source}" similarity="{item.similarity:.2f}">
<title>{item.title or "Untitled"}</title>
<url>{item.url}</url>
{f"<updated>{item.date}</updated>" if item.date else ""}
<content>
{item.text}
</content>
</document>"""
        parts.append(xml)

    return "\n\n".join(parts)


# =============================================================================
# MAIN CONTEXT OPTIMIZER
# =============================================================================


class ContextOptimizer:
    """Optimize context between reranker and generator."""

    def __init__(
        self,
        embedder: Callable | None = None,
        dedup_threshold: float = 0.85,
        max_per_source: int = 2,
        prefer_recent: bool = False,
    ):
        """Initialize context optimizer.

        Args:
            embedder: Optional embedding function for semantic dedup
            dedup_threshold: Similarity threshold for deduplication
            max_per_source: Max items from same source
            prefer_recent: Whether to prioritize recent content

        """
        self.embedder = embedder
        self.dedup_threshold = dedup_threshold
        self.max_per_source = max_per_source
        self.prefer_recent = prefer_recent

    def optimize(
        self,
        retrieval_results: list[dict],
        scores: list[float],
        query_intent: str | None = None,
        target_k: int = 5,
    ) -> list[dict]:
        """Optimize context for generator.

        Args:
            retrieval_results: Retrieved documents
            scores: Retrieval/reranking scores
            query_intent: Optional intent for ordering strategy
            target_k: Target number of context items

        Returns:
            Optimized documents for generator

        """
        # Convert to ContextItems
        items = [
            ContextItem.from_retrieval_result(doc, score)
            for doc, score in zip(retrieval_results, scores)
        ]

        # 1. Semantic deduplication
        items = deduplicate_semantic(items, self.embedder, self.dedup_threshold)

        # 2. Source diversity
        items = deduplicate_by_source(items, self.max_per_source)

        # 3. Ordering based on intent
        if query_intent == "procedural":
            items = order_procedural(items)
        elif self.prefer_recent:
            items = order_by_recency(items)
        else:
            items = order_by_type_priority(items)

        # 4. Limit to target
        items = items[:target_k]

        # 5. Augment with metadata
        return augment_with_metadata(items)

    def format_for_generator(
        self,
        retrieval_results: list[dict],
        scores: list[float],
        as_xml: bool = False,
    ) -> Any:
        """Format optimized context for generator.

        Args:
            retrieval_results: Retrieved documents
            scores: Retrieval scores
            as_xml: Whether to return XML string

        Returns:
            Formatted context (list of dicts or XML string)

        """
        items = [
            ContextItem.from_retrieval_result(doc, score)
            for doc, score in zip(retrieval_results, scores)
        ]

        if as_xml:
            return format_as_xml_context(items)
        else:
            return augment_with_metadata(items)
