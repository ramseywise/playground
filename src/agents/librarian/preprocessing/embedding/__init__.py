"""Embedding implementations.

- MultilingualEmbedder: intfloat/multilingual-e5-large (1024-dim, E5 prefix)
- MiniLMEmbedder: all-MiniLM-L6-v2 (384-dim, no prefix, English-only)
"""

from agents.librarian.preprocessing.embedding.embedders import (
    MiniLMEmbedder,
    MultilingualEmbedder,
)

__all__ = ["MiniLMEmbedder", "MultilingualEmbedder"]
