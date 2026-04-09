# Re-export from canonical location.
# Embedder implementations live in preprocessing/embedder.py;
# this shim keeps existing `retrieval.embedder` imports working.
from agents.librarian.preprocessing.embedder import MiniLMEmbedder, MultilingualEmbedder

__all__ = ["MultilingualEmbedder", "MiniLMEmbedder"]
