# Backward-compat shim — canonical location: preprocessing/chunking/html_aware.py
from agents.librarian.preprocessing.chunking.html_aware import HtmlAwareChunker
from agents.librarian.preprocessing.chunking.utils import (
    make_doc_id as _make_doc_id,
    recursive_split_by_separators as _recursive_split,
    word_count as _word_count,
)

__all__ = ["HtmlAwareChunker", "_make_doc_id", "_recursive_split", "_word_count"]
