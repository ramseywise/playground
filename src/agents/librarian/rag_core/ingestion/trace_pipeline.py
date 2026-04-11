"""Golden trace pipeline: extract -> ground -> store.

TODO(2): Implement the full pipeline once QAPairGenerator and TraceStore
have concrete implementations. Current scaffold defines the orchestration
contract and data flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.librarian.rag_core.schemas.traces import GoldenTrace, QAPair
from agents.librarian.utils.logging import get_logger

if TYPE_CHECKING:
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class TraceStore(Protocol):
        async def upsert_traces(self, traces: list[GoldenTrace]) -> None: ...
        async def upsert_qa_pairs(self, pairs: list[QAPair]) -> None: ...
        async def search_qa(self, query: str, k: int = 5) -> list[QAPair]: ...
        async def get_trace(self, trace_id: str) -> GoldenTrace | None: ...

    @runtime_checkable
    class QAPairGenerator(Protocol):
        async def generate(self, traces: list[GoldenTrace]) -> list[QAPair]: ...


log = get_logger(__name__)


async def run_trace_pipeline(
    documents: list[dict],
    generator: QAPairGenerator,
    store: TraceStore,
    *,
    trace_type: str = "text",
    text_field: str = "text",
) -> dict[str, int]:
    """Extract golden traces from documents, generate QA pairs, and store both.

    Args:
        documents: Raw document dicts with at least a text field.
        generator: QAPairGenerator implementation (e.g. LLM-based).
        store: TraceStore implementation (e.g. InMemory, DuckDB).
        trace_type: Default trace type tag.
        text_field: Field name containing text content.

    Returns:
        Dict with counts: {"traces": N, "qa_pairs": M}.

    """
    traces = _extract_traces(documents, trace_type=trace_type, text_field=text_field)
    if not traces:
        log.warning("trace_pipeline.no_traces")
        return {"traces": 0, "qa_pairs": 0}

    log.info("trace_pipeline.extracted", n_traces=len(traces))

    await store.upsert_traces(traces)

    qa_pairs = await generator.generate(traces)
    log.info("trace_pipeline.generated", n_qa_pairs=len(qa_pairs))

    await store.upsert_qa_pairs(qa_pairs)

    return {"traces": len(traces), "qa_pairs": len(qa_pairs)}


def _extract_traces(
    documents: list[dict],
    *,
    trace_type: str = "text",
    text_field: str = "text",
) -> list[GoldenTrace]:
    """Extract self-contained golden traces from documents.

    TODO(2): Add code-block extraction, config-block detection, and
    CLI command identification. Currently treats each document as a
    single trace.
    """
    import hashlib

    traces: list[GoldenTrace] = []
    for doc in documents:
        text = doc.get(text_field) or doc.get("content", "")
        if not text.strip():
            continue

        trace_id = hashlib.sha256(text[:256].encode()).hexdigest()[:16]
        traces.append(
            GoldenTrace(
                id=trace_id,
                text=text,
                source_url=doc.get("url", ""),
                source_title=doc.get("title", ""),
                language=doc.get("language", "en"),
                trace_type=trace_type,
                tags=doc.get("tags", []),
            )
        )

    return traces
