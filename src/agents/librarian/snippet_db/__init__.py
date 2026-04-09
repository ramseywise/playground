"""Snippet DB — grounded QA pairs pipeline.

Extracts code/text snippets from the corpus, generates grounded question-answer
pairs, and stores them for direct retrieval (no chunking needed).

Pipeline: extract snippets → generate QA pairs → ground answers → store/index

Submodules:
- base.py:     Protocols (SnippetStore, QAPairGenerator)
- models.py:   Pydantic schemas (Snippet, QAPair, GroundedAnswer)
- pipeline.py: End-to-end extract → ground → store orchestrator
"""
