# Librarian — Production RAG Service

A production-grade, config-driven RAG service built on LangGraph. Every component
(embedder, retriever, reranker, chunker) is swappable via environment variable. Runs
locally with Chroma + DuckDB, scales to ECS/Fargate with OpenSearch, and A/B tests
against AWS Bedrock Knowledge Bases from the same API surface.

> See [`.claude/docs/research/librarian-vs-bedrock-kb.md`](../../.claude/docs/research/librarian-vs-bedrock-kb.md)
> for a full tradeoff analysis: retrieval quality, latency, cost, and observability vs. Bedrock KB.

---

## Quick start

```bash
# Install
uv sync --extra librarian --extra api

cp .env.example .env
# Minimum required: ANTHROPIC_API_KEY=sk-ant-...

# Run the API
uv run librarian-api          # FastAPI on :8000

# Run the Streamlit chat UI
uv run streamlit run frontend/librarian_chat.py

# Run all tests
uv run pytest tests/librarian/
```

---

## System overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI  (tools/api/)                                              │
│  POST /api/v1/chat  ──► backend=librarian ──► LangGraph graph       │
│                     └─► backend=bedrock   ──► Bedrock KB (A/B)      │
│  POST /api/v1/ingest ──► IngestionPipeline                          │
│  GET  /health                                                        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ ainvoke({"query": ...})
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LangGraph state machine  (orchestration/graph.py)                  │
│                                                                     │
│  START → condense → analyze ──[direct]──────────────► generate      │
│                           └──[retrieve]──► retrieve                 │
│                                            ─► rerank                │
│                                            ─► gate ──[retry]──►┐   │
│                                            ─► generate ◄────────┘   │
│                                            ─► END                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## LangGraph orchestration

### State

All nodes read and write a single `LibrarianState` TypedDict. Every field is optional
(`total=False`) — nodes return partial dicts and LangGraph merges them.

```python
# pipeline/schemas/state.py
class LibrarianState(TypedDict, total=False):
    # Input
    query: str
    standalone_query: str        # condenser rewrites this for multi-turn
    conversation_id: str
    messages: Annotated[list[BaseMessage], add_messages]  # append-only

    # Planning
    intent: str                  # lookup | explore | compare | conversational | out_of_scope
    retrieval_mode: str          # dense | snippet
    query_variants: list[str]    # multi-query expansion

    # Retrieval
    retrieved_chunks: list[RetrievalResult]
    graded_chunks: list[GradedChunk]
    retry_count: int

    # Reranking
    reranked_chunks: list[RankedChunk]
    confidence_score: float

    # Generation
    response: str
    citations: list[dict]
    fallback_requested: bool
```

### Graph construction

```python
# orchestration/graph.py  (simplified)
def build_graph(
    retriever: Retriever,
    embedder: Embedder,
    reranker: Reranker,
    llm: LLMClient,
    *,
    history_condenser: HistoryCondenser | None = None,
    snippet_retriever: Retriever | None = None,
    cache: RetrievalCache | None = None,
    confidence_threshold: float = 0.3,
    max_crag_retries: int = 1,
) -> CompiledStateGraph:
    graph = StateGraph(LibrarianState)

    graph.add_node("condense",  _make_condense_node(condenser))
    graph.add_node("analyze",   _make_analyze_node(analyzer))
    graph.add_node("retrieve",  _make_retrieve_node(RetrievalSubgraph(...)))
    graph.add_node("rerank",    _make_rerank_node(RerankerSubgraph(...)))
    graph.add_node("gate",      _make_gate_node(GenerationSubgraph(...)))
    graph.add_node("generate",  _make_generate_node(GenerationSubgraph(...)))

    graph.add_edge(START, "condense")
    graph.add_edge("condense", "analyze")
    graph.add_conditional_edges("analyze", _route_after_analyze, {...})
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "gate")
    graph.add_conditional_edges("gate", _route_after_gate, {...})  # CRAG loop
    graph.add_edge("generate", END)

    return graph.compile()
```

### Node types

Every node factory returns a typed callable — sync for CPU-only nodes, async for I/O:

```python
_SyncNode  = Callable[[LibrarianState], dict[str, Any]]
_AsyncNode = Callable[[LibrarianState], Coroutine[Any, Any, dict[str, Any]]]
```

### CRAG retry loop

The `gate` node checks `confidence_score` from the reranker. Below threshold, it sets
`fallback_requested=True` and increments `retry_count`. `_route_after_gate` sends the
graph back to `retrieve` with a reformulated query — up to `max_crag_retries` times,
then generates from whatever context is available.

```python
def _route_after_gate(state, max_retries) -> Literal["generate", "retrieve"]:
    if state.get("fallback_requested") and state.get("retry_count", 0) <= max_retries:
        return "retrieve"   # CRAG retry
    return "generate"
```

### Multi-turn condensation

The `condense` node runs before every query. For single-turn it's a no-op. For
multi-turn it calls Haiku to rewrite the latest message as a self-contained query,
resolving coreference ("what about the Python one?") before retrieval sees it.

```python
# orchestration/history.py
class HistoryCondenser:
    async def condense(self, state: LibrarianState) -> dict[str, Any]:
        messages = state.get("messages", [])
        if len(messages) <= 1:
            return {"standalone_query": state.get("query", "")}
        # Haiku call: rewrite latest message given prior turns
        rewritten = await self._llm.generate(CONDENSE_SYSTEM, history_messages)
        return {"standalone_query": rewritten}
```

---

## Customisable RAG pipeline

Every stage is a Protocol. Swap implementations via env var — no code change.

### Protocols

```python
# pipeline/retrieval/base.py
class Embedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...    # adds "query: " prefix
    def embed_passage(self, text: str) -> list[float]: ...  # adds "passage: " prefix
    async def aembed_query(self, text: str) -> list[float]: ...
    async def aembed_passages(self, texts: list[str]) -> list[list[float]]: ...

class Retriever(Protocol):
    async def search(self, query_text: str, query_vector: list[float],
                     k: int = 10, metadata_filter: dict | None = None,
                     ) -> list[RetrievalResult]: ...
    async def upsert(self, chunks: list[Chunk]) -> None: ...

# pipeline/reranker/base.py
class Reranker(Protocol):
    async def rerank(self, query: str, chunks: list[GradedChunk],
                     top_k: int = 3) -> list[RankedChunk]: ...

# pipeline/ingestion/base.py
class Chunker(Protocol):
    def chunk_document(self, doc: dict) -> list[Chunk]: ...
```

### Factory — config-driven DI

`create_librarian()` is the single entry point. Pass overrides for testing; let config
drive everything in production.

```python
# factory.py
def create_librarian(
    cfg: LibrarySettings | None = None,
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
) -> CompiledStateGraph:
    cfg = cfg or settings
    return build_graph(
        retriever = retriever or _build_retriever(cfg, embedder),
        embedder  = embedder  or _build_embedder(cfg),
        reranker  = reranker  or _build_reranker(cfg, llm),
        llm       = llm       or _build_llm(cfg),
        cache     = RetrievalCache(...) if cfg.cache_enabled else None,
        ...
    )
```

### Strategy dispatch table

| Config var | Values | Default |
|---|---|---|
| `RETRIEVAL_STRATEGY` | `chroma` · `opensearch` · `duckdb` · `inmemory` | `chroma` |
| `RERANKER_STRATEGY` | `cross_encoder` · `llm_listwise` · `passthrough` | `cross_encoder` |
| `INGESTION_STRATEGY` | `html_aware` · `parent_doc` · `fixed` · `overlapping` · `structured` · `adjacency` | `html_aware` |
| `EMBEDDING_PROVIDER` | `multilingual` · `minilm` | `multilingual` |
| `PLANNING_MODE` | `rule_based` · `llm` | `rule_based` |

### Retrieval internals

**Hybrid scoring:** Each retriever combines BM25 term-overlap and cosine similarity
with configurable weights (`BM25_WEIGHT=0.3`, `VECTOR_WEIGHT=0.7`).

**Multi-query + RRF:** The retrieval node expands the query into up to N variants via
`QueryAnalyzer`, retrieves in parallel with `asyncio.gather`, then fuses ranked lists
using Reciprocal Rank Fusion (`pipeline/retrieval/rrf.py`):

```python
def fuse_rankings(result_lists: list[list[RetrievalResult]], k: int = 60) -> list[RetrievalResult]:
    """RRF: score(d) = Σ 1/(k + rank(d, list_i)) across all lists."""
```

**Cache:** `RetrievalCache` is a thread-safe TTL LRU cache (default: 256 entries,
5-minute TTL) keyed on `(query, strategy, top_k)`. Injected via factory, disabled
in tests.

---

## Ingestion pipeline

```python
# Ingest a batch of documents
pipeline = create_ingestion_pipeline()

docs = [{"url": "...", "title": "...", "text": "..."}]
result = await pipeline.run(docs)
# → chunks stored in VectorDB, metadata in DuckDB, snippets in SnippetDB
```

**Pipeline stages:**
1. **Parse** — clean HTML, detect language, deduplicate, enrich metadata
2. **Chunk** — strategy-selected chunker (default: HTML-aware heading-boundary recursive split)
3. **Embed** — `MultilingualEmbedder` with E5 prefix enforcement (`"passage: "` prefix on index)
4. **Index** — upsert to VectorDB; write `ChunkMetadata` to MetadataDB; write FTS snippets to SnippetDB

---

## Eval system

```
tests/librarian/
  unit/           → 355 tests, no LLM calls, no Docker, MockEmbedder + InMemoryRetriever
  evalsuite/
    regression/   → hit_rate@5 ≥ 0.6, MRR ≥ 0.4  (CI gate — never lower these floors)
    capability/   → end-to-end routing, CRAG termination, multi-turn, streaming
```

### Eval architecture

```
GoldenSample → retrieve_fn → RetrievalMetrics + FailureCluster[]
             ↓
             AnswerJudge (LLM-as-judge, guarded by CONFIRM_EXPENSIVE_OPS)
             → JudgeResult(is_correct, faithfulness, relevance, completeness)
```

**Golden dataset tiers:**
- `gold` — hand-curated with expected chunk IDs
- `silver` — human-validated query/URL pairs
- `bronze` — inferred from interaction logs

**Failure clustering** groups misses by root cause:
```
zero_retrieval            → nothing came back (corpus gap or embedding failure)
expected_doc_not_in_top_k → right doc exists but wasn't ranked high enough
low_confidence            → retrieved but reranker didn't trust it
wrong_intent              → classified incorrectly, wrong retrieval path
```

**Run targets:**
```bash
uv run pytest tests/librarian/unit/                           # fast, always
uv run pytest tests/librarian/evalsuite/regression/           # CI gate
uv run pytest tests/librarian/evalsuite/capability/           # end-to-end
CONFIRM_EXPENSIVE_OPS=true uv run pytest -k answer_judge      # LLM-as-judge
```

---

## API surface

```
POST /api/v1/chat
  Body: { "query": str, "session_id": str, "backend": "librarian"|"bedrock" }
  Response: { "response": str, "citations": [...], "confidence_score": float,
              "intent": str, "backend": str, "trace_id": str }

POST /api/v1/chat/stream           (librarian backend only)
  Body: same as /chat
  Response: SSE stream of response tokens

POST /api/v1/ingest
  Body: { "documents": [{"url", "title", "text"}] }
  Response: { "ingested": int, "failed": int, "results": [...] }

GET  /api/v1/health
GET  /health                       (ALB/ECS health check)
```

The `backend` field enables live A/B comparison between the custom pipeline and
AWS Bedrock Knowledge Bases without any frontend change.

---

## Deployment: ECS/Fargate

### Recommended task spec

```hcl
# infra/terraform/variables.tf
cpu    = 2048   # 2 vCPU
memory = 4096   # 4 GB — required for multilingual-e5-large (~560MB) + runtime

# Health check
startPeriod = 60  # model loads ~45s on cold start
```

### Expected latency (warm task)

| Stage | Time |
|---|---|
| Query condense (single-turn no-op) | ~1ms |
| Intent classify (rule-based) | ~5ms |
| Embed query (e5-large, warm) | ~100–200ms |
| Chroma search | ~50ms |
| Cross-encoder rerank (MiniLM, warm) | ~200–500ms |
| Claude Sonnet on Bedrock TTFT | ~400–800ms |
| **Total to first streamed token** | **~800ms–1.5s** |

### Reducing cold-start penalty

The embedding model loads lazily on first request. To warm it at startup, hit
`/api/v1/health` with a lightweight embed after container start, or replace the local
model with a cloud embedding API (`EMBEDDING_PROVIDER=voyage`) — removes the 560MB
model entirely and cuts embed latency to ~50ms.

### Bedrock model (production)

Swap the personal Anthropic API key for a Bedrock model ARN — no code change:

```bash
# .env
BEDROCK_MODEL_ARN=arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6-20251001-v2:0
BEDROCK_REGION=us-east-1
# Leave ANTHROPIC_API_KEY unset
```

The `AnthropicLLM` client in `tools/core/clients/llm.py` handles both paths.

### Optional: Bedrock Knowledge Bases (A/B baseline)

```bash
# .env — activates the bedrock backend in the UI
BEDROCK_KNOWLEDGE_BASE_ID=<kb-id>
```

When set, the Streamlit UI exposes a radio button to toggle between the custom
pipeline and Bedrock KB on the same query. See the research doc for a full comparison.

---

## Project structure

```
src/librarian/                     (symlinked as agents/librarian/)
├── factory.py                     create_librarian(), create_ingestion_pipeline()
│
├── orchestration/
│   ├── graph.py                   build_graph() → CompiledStateGraph
│   ├── history.py                 HistoryCondenser (multi-turn query rewrite)
│   ├── query_understanding.py     QueryAnalyzer, QueryRouter (re-export shim)
│   └── nodes/
│       ├── retrieval.py           RetrievalSubgraph (embed → search → grade → RRF)
│       ├── reranker.py            RerankerSubgraph (cross-encoder → confidence)
│       └── generation.py         GenerationSubgraph (prompt → LLM → citations → gate)
│
├── pipeline/
│   ├── schemas/                   LibrarianState, Chunk, RetrievalResult, QueryPlan
│   ├── plan/                      analyzer, routing, intent, expansion, decomposition
│   ├── ingestion/                 base (Chunker Protocol), chunking/, parsing/,
│   │                              indexing/, embeddings/, pipeline.py
│   ├── retrieval/                 base (Embedder/Retriever Protocols), cache, rrf, scoring
│   ├── reranker/                  base (Reranker Protocol), cross_encoder, llm_listwise
│   ├── generation/                generator, prompts, context
│   └── bedrock/                   BedrockKBClient (A/B baseline, optional)
│
├── tools/
│   ├── api/                       FastAPI app, routes, deps, middleware, lambda_handler
│   ├── mcp/                       MCP servers: librarian, S3, Snowflake
│   ├── storage/
│   │   ├── vectordb/              chroma, opensearch, duckdb, inmemory
│   │   ├── metadatadb/            DuckDB-backed chunk metadata store
│   │   ├── tracedb/               DuckDB-backed conversation trace store
│   │   └── graphdb/               (future: knowledge graph)
│   └── core/                      LLMClient, BaseSettings, storage protocols
│
├── eval/
│   ├── datasets/                  GoldenSample store, SnippetStore
│   ├── graders/                   AnswerJudge, ExactMatch, MCQ, DeepEval, RAGAS
│   ├── metrics/                   evaluate_retrieval() → RetrievalMetrics + FailureCluster
│   ├── tasks/                     extract_golden, generate_synthetic, tracing
│   └── runner.py                  EvalRunner orchestration
│
└── utils/
    ├── config.py                  LibrarySettings (pydantic-settings, all env vars)
    ├── llm.py                     AnthropicLLM (direct SDK + Bedrock model ARN)
    ├── logging.py                 structlog get_logger
    ├── otel.py                    OpenTelemetry setup (optional)
    └── tracing.py                 trace helpers

tests/librarian/
├── testing/mock_embedder.py       MockEmbedder (deterministic, seed-stable)
├── unit/                          355 tests — all pass without Docker or API keys
└── evalsuite/                     regression floors + capability end-to-end
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Personal Claude key (dev only) |
| `BEDROCK_MODEL_ARN` | — | Bedrock model ARN (production) |
| `BEDROCK_REGION` | — | AWS region for Bedrock |
| `BEDROCK_KNOWLEDGE_BASE_ID` | — | Activates Bedrock KB A/B toggle |
| `RETRIEVAL_STRATEGY` | `chroma` | `chroma` · `opensearch` · `duckdb` · `inmemory` |
| `RERANKER_STRATEGY` | `cross_encoder` | `cross_encoder` · `llm_listwise` · `passthrough` |
| `INGESTION_STRATEGY` | `html_aware` | `html_aware` · `parent_doc` · `fixed` · `overlapping` |
| `EMBEDDING_PROVIDER` | `multilingual` | `multilingual` · `minilm` |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | SentenceTransformer model name |
| `CONFIDENCE_THRESHOLD` | `0.4` | CRAG gate — retry retrieval below this |
| `MAX_CRAG_RETRIES` | `1` | Max CRAG retry iterations |
| `RETRIEVAL_K` | `10` | Candidates fetched from vector store |
| `RERANKER_TOP_K` | `3` | Chunks passed to generation |
| `BM25_WEIGHT` | `0.3` | Hybrid search BM25 blend |
| `VECTOR_WEIGHT` | `0.7` | Hybrid search vector blend |
| `CACHE_ENABLED` | `true` | Enable retrieval result cache |
| `CACHE_TTL_SECONDS` | `300` | Cache TTL |
| `CHROMA_PERSIST_DIR` | `.chroma` | Local Chroma persistence path |
| `DUCKDB_PATH` | `.duckdb/librarian.db` | Shared DuckDB path (metadata + traces) |
| `OPENSEARCH_URL` | `http://localhost:9200` | OpenSearch endpoint |
| `API_CORS_ORIGINS` | `http://localhost:3000,http://localhost:8501` | Allowed CORS origins |
| `LANGFUSE_ENABLED` | `false` | Enable LangFuse trace export |
| `CONFIRM_EXPENSIVE_OPS` | `false` | Gate for LLM-as-judge + synthetic eval |
