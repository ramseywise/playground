# Preprocessing (offline / batch)

This package is the **index-building** side of RAG: ingest documents, normalize text, chunk, optionally enrich, and write embeddings into your chosen vector store / metadata DB. It runs **outside** the request path (batch jobs, local demos, CI fixtures).

| Concern | Where to look |
|---------|----------------|
| Chunking, parsing, ingestion pipeline | `pipeline.py`, `chunking/`, `parsing/`, `loaders.py` |
| CLI: chunk + embed + upsert Markdown | `ingest_cli.py` — e.g. `uv run python -m app.rag.preprocessing.ingest_cli --directory data/document` (see module docstring for env vars) |

## Online vs offline

- **Offline (this directory):** build or refresh the **corpus index** — crawl/export → preprocess → index. Tuning chunk size and metadata here affects **what** gets retrieved later, not LangGraph routing.
- **Online (query time):** `app/rag/retrieval/runtime.py` (local retriever factory), `app/rag/retrieval/pipeline.py` (ensemble retrieve + rerank steps used by graph nodes), plus `app/rag/embedding/` for query embeddings. That path answers **“what matches this user query right now?”**

Production MVP in this repo’s product plan uses **Amazon Bedrock Knowledge Base** ingestion and sync instead of (or in addition to) this local pipeline — see [Phase B — Ingestion & index](../../../.claude/docs/plans/rag-system-development-plan.md#phase-b--ingestion--index) in the development plan.
