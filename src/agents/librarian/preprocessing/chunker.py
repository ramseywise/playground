# Backward-compat shim — canonical location: preprocessing/chunking/
from agents.librarian.preprocessing.chunking.strategies import (
    AdjacencyChunker,
    FixedChunker,
    OverlappingChunker,
    StructuredChunker,
    _split_sections,
)
from agents.librarian.preprocessing.chunking.utils import (
    approx_tokens as _approx_tokens,
    hard_split_text as _hard_split_text,
    make_chunk as _make_chunk,
    make_doc_id as _make_doc_id,
    merge_with_overlap as _merge_with_overlap,
    recursive_split_with_config as _recursive_split,
)

__all__ = [
    "AdjacencyChunker",
    "FixedChunker",
    "OverlappingChunker",
    "StructuredChunker",
    "_approx_tokens",
    "_hard_split_text",
    "_make_chunk",
    "_make_doc_id",
    "_merge_with_overlap",
    "_recursive_split",
    "_split_sections",
]
