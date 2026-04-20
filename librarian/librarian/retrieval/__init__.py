"""Retrieval subsystem — infrastructure backends, protocols, and test utilities.

Subpackages:
- infra/    — Backend clients (ChromaRetriever, OpenSearchRetriever, DuckDBRetriever, InMemoryRetriever)
- testing/  — Test utilities (MockEmbedder)
- base.py   — Embedder + Retriever Protocols
- ensemble.py — Multi-query, multi-retriever fusion with fingerprint dedup
"""
