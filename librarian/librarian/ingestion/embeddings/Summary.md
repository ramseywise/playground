Assessment Summary
rag_poc-master strengths (things librarian doesn't have):

Human-in-the-loop interrupts (clarify + confirm nodes)
Multi-intent routing (Q&A vs task execution)
Clean, readable 6-node graph — easy starting point
Critical gaps to close (in priority order):

Gap	Why It Matters
No reranking	All retrieved docs treated equally — retrieval quality is low
No CRAG loop	Single failed retrieval = bad answer, no recovery
No query expansion	One embedding vector misses paraphrased queries
In-memory only vector store	Loses all data on restart
No confidence gating	Can't distinguish confident answers from hallucinations
Hardcoded OpenAI + globals	Can't swap models, can't inject alternatives, can't test
No structured config (Pydantic)	No validation, no env-var override
No observability	print() only — production debugging is blind
No tests	Any refactor is a gamble
Incomplete (no main.py)	Dockerfile CMD references app.main:app which doesn't exist
Scaffolded Git Commit Plan
Organized into 5 phases — each phase is a mergeable PR milestone with atomic commits. The rule: each commit leaves the app in a runnable state.

Phase 0 — Foundation Fix (1 PR, 3 commits)
Make the POC actually run before building on it.


feat: add FastAPI entrypoint (app/main.py) with /chat and /health endpoints
fix: wire LangGraph graph to FastAPI — thread_id + message in, answer out
chore: add .env.example with all required env vars + dev requirements
Phase 1 — Config & Structure (1 PR, 4 commits)
Replace global singletons with Pydantic settings. Required before any DI work.


refactor: replace app/core/config.py globals with Pydantic BaseSettings
refactor: add LLM provider abstraction — AnthropicProvider + OpenAIProvider
refactor: wire settings into graph nodes via constructor injection (no more os.getenv)
chore: add pytest + conftest.py scaffold, first smoke test on /health

Phase 2 — Retrieval Upgrade (1 PR, 5 commits)
The biggest quality gap. Swap in-memory for persistent + add hybrid search.


feat: replace InMemoryVectorStore with ChromaDB (persist_directory configurable)
feat: add BM25 sparse retriever alongside dense + RRF fusion (k=60)
feat: add multi-query expansion — generate N query variants before retrieval
feat: add content deduplication on retrieved chunks (SHA-256 fingerprint)
test: unit tests for retrieval pipeline (dense, sparse, fusion, dedup)
Phase 3 — Reranking + Confidence Gating (1 PR, 4 commits)
The CRAG loop and quality signal. This is what makes answers trustworthy.


feat: add cross-encoder reranker (ms-marco-MiniLM-L-6-v2) with sigmoid normalization
feat: add confidence gating node — scores top-k, gates on threshold (default 0.4)
feat: add CRAG retry loop — re-retrieves with expanded query if confidence < threshold
test: unit tests for reranker + confidence gate; integration test for CRAG loop
Phase 4 — Observability & Response Schema (1 PR, 3 commits)
Structured output + production visibility.


feat: add RAGResponse schema — answer, citations, confidence_score, intent
feat: add structured logging (structlog) + request/response log middleware
feat: add OpenTelemetry tracing (Phoenix or OTLP) — instrument LangGraph nodes + LLM calls
Phase 5 — Production Hardening (1 PR, 4 commits)
Multi-worker checkpointing, embedder swap, final test coverage.


feat: add SQLite checkpointer (dev) + Postgres checkpointer (prod) configurable via env
feat: add HuggingFace embedder option (E5-large or MiniLM) alongside OpenAI
feat: add retrieval cache (TTL LRU, configurable max_size + ttl_seconds)
test: integration tests for full RAG pipeline; happy path + CRAG retry path
What Stays from rag_poc-master
These are unique to the POC — keep them, librarian doesn't have them:

Human-in-the-loop clarify/confirm nodes (they're good)
Multi-intent routing (Q&A vs task execution)
SchedulerOutput plan generation
These get refactored, not rewritten — after Phase 1 they'll use injected LLMs instead of hardcoded globals.

Suggested branch naming (LIN-linked)

phase/0-foundation-fix
phase/1-pydantic-config
phase/2-retrieval-upgrade
phase/3-reranker-crag
phase/4-observability
phase/5-prod-hardening