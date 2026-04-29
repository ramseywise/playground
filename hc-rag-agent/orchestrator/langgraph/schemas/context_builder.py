"""Build answer context from ranked chunks with token budget and deduplication."""

from __future__ import annotations

from dataclasses import dataclass

from rag.schemas.chunks import RankedChunk
from rag.tokenization import count_tokens


@dataclass(frozen=True)
class ContextBuildConfig:
    """Limits for assembling retrieved text fed to the answer LLM."""

    max_tokens: int = 6000
    max_chunks: int = 12
    dedupe_by_chunk_id: bool = True


@dataclass(frozen=True)
class ContextBuildResult:
    """Formatted context block plus which chunks were included (for citations)."""

    text: str
    chunk_ids_in_order: tuple[str, ...]
    truncated: bool
    tokens_used: int


def _format_one(rc: RankedChunk) -> str:
    m = rc.chunk.metadata
    src = m.title or m.url or m.doc_id or rc.chunk.id
    return (
        f"Rank {rc.rank} (relevance {rc.relevance_score:.4f}):\n"
        f"Source: {src}\n"
        f"Content:\n{rc.chunk.text}\n"
    )


def build_context_from_ranked(
    ranked: list[RankedChunk],
    config: ContextBuildConfig | None = None,
) -> ContextBuildResult:
    """Select and format ranked chunks up to *max_tokens* and *max_chunks*.

    Chunks are visited in **rank order**. Duplicate *chunk ids* are skipped when
    *dedupe_by_chunk_id* is true. Matches the visual style of
    :func:`app.graph.schemas.contract.format_reranked_context` for included passages only.
    """
    cfg = config or ContextBuildConfig()
    if not ranked:
        return ContextBuildResult(
            text="No relevant documents found.",
            chunk_ids_in_order=(),
            truncated=False,
            tokens_used=0,
        )

    ordered = sorted(ranked, key=lambda r: r.rank)
    seen: set[str] = set()
    lines: list[str] = []
    ids: list[str] = []
    truncated = False

    for rc in ordered:
        if cfg.dedupe_by_chunk_id:
            cid = rc.chunk.id
            if cid in seen:
                continue
            seen.add(cid)
        if len(ids) >= cfg.max_chunks:
            truncated = True
            break
        block = _format_one(rc)
        trial_lines = lines + [block]
        trial = "\n" + "=" * 60 + "\n" + "\n".join(trial_lines)
        if count_tokens(trial) > cfg.max_tokens:
            truncated = True
            break
        lines = trial_lines
        ids.append(rc.chunk.id)

    if not lines:
        return ContextBuildResult(
            text="No relevant documents found.",
            chunk_ids_in_order=(),
            truncated=True,
            tokens_used=0,
        )

    text = "\n" + "=" * 60 + "\n" + "\n".join(lines)
    return ContextBuildResult(
        text=text,
        chunk_ids_in_order=tuple(ids),
        truncated=truncated,
        tokens_used=count_tokens(text),
    )


__all__ = [
    "ContextBuildConfig",
    "ContextBuildResult",
    "build_context_from_ranked",
    "count_tokens",
]
