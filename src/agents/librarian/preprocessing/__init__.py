"""Preprocessing subsystem — chunking, embedding, indexing, and parsing.

Subpackages:
- chunking/   — Chunking strategies (HtmlAwareChunker, ParentDocChunker, FixedChunker, etc.)
- embedding/  — Embedder implementations (MultilingualEmbedder, MiniLMEmbedder)
- indexing/   — Index pipeline (ChunkIndexer, build_indexer_for_source)
- parsing/    — Document parsing (clean_text, dedup, language detection, enrichment)
- base.py     — ChunkerConfig + Chunker Protocol
"""
