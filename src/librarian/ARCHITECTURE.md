# Librarian Architecture

> RAG agent built on LangGraph with CRAG (Corrective RAG) loop, confidence gating,
> and fully config-driven component injection.

---

## 1. State Machine

```
                    condense
                       │
                       ▼
                    analyze
                       │
              ┌────────┴────────┐
              ▼                 ▼
          retrieve       snippet_retrieve
              │                 │
              ▼                 │
           rerank               │
              │                 │
              ▼                 │
            gate                │
              │                 │
       ┌──────┴──────┐         │
       ▼             ▼         │
   retrieve       generate ◄───┘
  (CRAG retry)       │
                     END
```

**Node responsibilities:**

| Node | Module | What it does |
|------|--------|-------------|
| `condense` | `orchestration/history.py` | Rewrites multi-turn queries to standalone form (Haiku). No-op on single-turn. |
| `analyze` | `orchestration/query_understanding.py` | Rule-based intent classification + query expansion + entity extraction. |
| `retrieve` | `orchestration/nodes/retrieval.py` | Multi-query expansion → parallel embedding → hybrid search → deduplicate. |
| `snippet_retrieve` | `graph.py` (inline) | Keyword-only DuckDB FTS path for simple factual lookups. Bypasses embedder + reranker. |
| `rerank` | `orchestration/nodes/reranker.py` | Cross-encoder or LLM listwise reranking → `confidence_score`. |
| `gate` | `orchestration/nodes/generation.py` | Compares `confidence_score` to threshold → `confident` / `fallback_requested`. |
| `generate` | `orchestration/nodes/generation.py` | LLM call with system prompt + reranked context → `response` + `citations`. |


## 2. CRAG Loop

The Corrective RAG loop fires when the reranker's confidence is below threshold:

1. **Gate** evaluates `confidence_score >= confidence_threshold`
2. If below AND `retry_count <= max_crag_retries`: route back to `retrieve`
3. `retry_count` is incremented on each fallback pass
4. After exhausting retries: route to `generate` regardless (graceful degradation)

**Config:**
- `confidence_threshold` (default: 0.4) — minimum reranker max-score to proceed
- `max_crag_retries` (default: 1) — how many extra retrieval attempts


## 3. Confidence Gating

```
reranker output → max(relevance_score across reranked_chunks) → confidence_score
confidence_score >= threshold → confident=True  → generate
confidence_score <  threshold → confident=False → fallback_requested=True → CRAG retry
```

- `confidence_score` = maximum `relevance_score` from the reranker's top-k results
  - Cross-encoder: raw logit passed through sigmoid
  - LLM listwise: normalised position score
- When no chunks pass reranking: `confidence_score = 0.0`
- The gate is a **separate node** (not inside generate) so the CRAG conditional edge can read its output


## 4. DI Wiring (`factory.py`)

`factory.py` is the **single assembly point** for all components. Strategy selection is config-driven via `LibrarySettings` (pydantic-settings, env-var overridable).

**Component swap points:**

| Component | Config field | Options |
|-----------|-------------|---------|
| Embedder | `embedding_provider` | `multilingual` (default), `minilm` |
| Retriever | `retrieval_strategy` | `chroma` (default), `opensearch`, `duckdb`, `inmemory` |
| Reranker | `reranker_strategy` | `cross_encoder` (default), `llm_listwise`, `passthrough` |
| LLM | `llm_provider` | `anthropic` (default), `gemini` |
| Chunker | `ingestion_strategy` | `html_aware` (default), `fixed`, `overlapping`, `structured`, `adjacency`, `parent_doc` |
| Checkpointer | `checkpoint_backend` | `memory` (default), `sqlite`, `postgres` |

Any component can also be injected directly via `create_librarian(embedder=...)` — this is the primary pattern in tests.

**Graph construction flow:**
```
create_librarian(cfg)
  → _build_llm(cfg)             → AnthropicLLM | GeminiLLM
  → _build_history_llm(cfg)     → Haiku for condenser
  → _build_embedder(cfg)        → MultilingualEmbedder | MiniLMEmbedder
  → _build_retriever(cfg, emb)  → ChromaRetriever | OpenSearchRetriever | ...
  → _build_reranker(cfg, llm)   → CrossEncoderReranker | LLMListwiseReranker
  → _build_checkpointer(cfg)    → MemorySaver | SqliteSaver | PostgresSaver
  → build_graph(all components) → CompiledStateGraph
```


## 5. Key Invariants

1. **E5 prefix rule**: `MultilingualEmbedder` prepends `"query: "` for search-time and `"passage: "` for index-time. All retrieval code must use `embed_query` / `embed_passage` — never `encode()` directly.

2. **Chroma single-writer**: `PersistentClient` holds a process-level write lock. Within-process upserts are serialised via `_WRITE_LOCK` (asyncio.Lock). Multi-worker ingest requires `retrieval_strategy=opensearch`.

3. **Embedder lazy load**: `_load_model()` is lazy — the 560MB model loads on first `embed_query`. Call `warm_up_embedder()` in the API lifespan to avoid cold-start latency. The `_MODEL_CACHE` is process-wide and keyed by `model_name@revision`.

4. **Model version pinning**: Set `EMBEDDING_MODEL_REVISION` to a HuggingFace commit SHA to prevent silent model drift. When unset, `SentenceTransformer` downloads the latest revision.


## 6. Checkpointer Model

The graph is compiled with a checkpointer that persists conversation state keyed by `thread_id`.

- **thread_id**: derived from `session_id` in the API request. If not provided, a UUID is generated per request.
- **Memory** (default): In-process dict — state is lost on restart. Suitable for single-turn or dev.
- **SQLite**: File-backed — survives process restarts but not task migration.
- **Postgres**: Network-backed — survives deploys, scale events, and task restarts. Required for production multi-turn.

Callers must pass `config={"configurable": {"thread_id": "..."}}` to `graph.ainvoke()` / `graph.astream()` when a checkpointer is active.
