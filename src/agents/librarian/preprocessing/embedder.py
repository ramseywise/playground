# Backward-compat shim — canonical location: preprocessing/embedding/embedders.py
from agents.librarian.preprocessing.embedding.embedders import (
    MiniLMEmbedder,
    MultilingualEmbedder,
)

__all__ = ["MultilingualEmbedder", "MiniLMEmbedder"]
