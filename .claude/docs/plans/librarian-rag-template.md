# Plan — Librarian Agent: Multi-Source RAG Template

> Target: `src/agents/librarian/` + `tests/librarian/`
> Scope: `src/` + unit tests only — no API, no infra, no frontend.
> Research: `.claude/docs/research/rag-agent-template.md`
> Date: 2026-04-09

---

## Overview

Build a supervisor multi-agent RAG system as a reusable template for search and retrieval
over any multi-source corpus — documentation, wikis, structured records, code, or mixed
collections. Four conceptual agents (Planning, Retrieval, Reranking, Generation) implemented
as LangGraph subgraphs coordinated by a supervisor. All strategies are registry-swappable
via env var for A/B experimentation.

```
supervisor_graph
  ├── plan_node            ← intent classification + routing decision
  ├── retrieval_subgraph   ← rewrite → multi-query → hybrid search → CRAG grade
  ├── reranker_subgraph    ← score chunks → filter top-k → set confidence_score
  ├── generation_subgraph  ← build prompt → call LLM → extract citations
  └── confidence_gate      ← threshold check → fallback or return response
```

Dependencies between steps are noted. Steps within the same phase can be done in any order
unless an arrow indicates otherwise.

---

## Step 1 — Package scaffold + pyproject.toml

**Files to create:**

```
src/agents/librarian/__init__.py
src/agents/librarian/schemas/__init__.py
src/agents/librarian/ingestion/__init__.py
src/agents/librarian/retrieval/__init__.py
src/agents/librarian/reranker/__init__.py
src/agents/librarian/generation/__init__.py
src/agents/librarian/orchestration/__init__.py
src/agents/librarian/orchestration/subgraphs/__init__.py
src/agents/librarian/evals/__init__.py
src/agents/librarian/utils/__init__.py
tests/librarian/__init__.py
tests/librarian/unit/__init__.py
tests/librarian/unit/conftest.py
```

`conftest.py` is created here as a stub with the `reset_registry` autouse fixture and placeholder comments for `MockEmbedder`, `InMemoryRetriever`, and mock LLM fixtures. These are populated in Step 4 (retrieval). All subsequent steps assume conftest is fully populated.

**`pyproject.toml` additions** (confirm before touching):

```toml
[project.optional-dependencies]
library = [
    "langgraph>=0.4.0",
    "langchain-core>=0.3.0",
    "langchain-anthropic>=0.3.0",
    "opensearch-py>=2.7.0",
    "sentence-transformers>=3.0.0",
    "ragas>=0.2.0",
    "deepeval>=2.0.0",
    "langfuse>=2.0.0",
]
```

Install: `uv add --optional library langgraph langchain-core langchain-anthropic opensearch-py sentence-transformers ragas deepeval langfuse`

**No code yet — scaffold only.**

---

## Step 2 — Schemas

**Files:**

- `src/agents/librarian/schemas/chunks.py`
- `src/agents/librarian/schemas/retrieval.py`
- `src/agents/librarian/schemas/state.py`
- `tests/librarian/unit/test_schemas.py`

### `chunks.py`

```python
class ChunkMetadata(BaseModel):
    # Core (always required)
    url: str
    title: str
    doc_id: str
    # Optional — present in structured corpora
    section: str | None = None
    language: str = "en"           # ISO 639-1; multilingual-e5-large supports 100+ languages
    parent_id: str | None = None   # parent_doc strategy
    # Corpus-specific fields — None by default, additive when needed
    namespace: str | None = None   # corpus partition, e.g. "docs", "wiki", "api-reference"
    topic: str | None = None       # subject area, e.g. "authentication", "billing", "setup"
    content_type: str | None = None  # "article", "tutorial", "reference", "changelog"
    access_tier: str | None = None   # access control: "public", "internal", "premium"
    last_updated: str | None = None  # ISO 8601, freshness detection
    source_id: str | None = None     # upstream record ID from ingestion source
    completeness_score: float | None = None  # quality gate at ingestion

class Chunk(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None

class GradedChunk(BaseModel):
    chunk: Chunk
    score: float          # retrieval score
    relevant: bool        # CRAG relevance judgment

class RankedChunk(BaseModel):
    chunk: Chunk
    relevance_score: float   # reranker output (0–1)
    rank: int
```

### `retrieval.py`

```python
class Intent(str, Enum):
    LOOKUP = "lookup"           # find a specific fact or record
    EXPLORE = "explore"         # open-ended investigation across sources
    COMPARE = "compare"         # side-by-side of options or versions
    CONVERSATIONAL = "conversational"  # greetings, clarifications, chitchat
    OUT_OF_SCOPE = "out_of_scope"      # outside the corpus domain

class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    source: Literal["vector", "bm25", "hybrid"]

class QueryPlan(BaseModel):
    intent: Intent
    routing: Literal["retrieve", "direct", "clarify"]
    query_variants: list[str]        # multi-query expansion
    needs_clarification: bool
    clarification_question: str | None = None
```

### `state.py`

```python
class LibrarianState(TypedDict, total=False):
    # Core
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    standalone_query: str
    trace_id: str

    # Planning output
    intent: str
    plan: QueryPlan
    skip_retrieval: bool

    # Retrieval output
    query_variants: list[str]
    retrieved_chunks: list[RetrievalResult]
    graded_chunks: list[GradedChunk]
    retry_count: int           # CRAG loop counter

    # Reranker output
    reranked_chunks: list[RankedChunk]
    confidence_score: float    # max relevance_score from reranker

    # Generation output
    response: str
    citations: list[dict]      # [{"url": ..., "title": ...}]
    confident: bool            # confidence_gate result
    fallback_requested: bool   # set True when confidence_gate fires
```

**`messages` vs all other fields:** `messages` uses the `add_messages` reducer — nodes that return `{"messages": [...]}` _append_ to the list. All other `LibrarianState` fields are _replace-on-update_ (last writer wins). Tests must account for this: asserting `state["messages"][-1]` not `state["messages"][0]`.

**Test:** validate Pydantic models reject bad input; TypedDict field access works.

---

## Step 2.5 — Utils

**Must run before Step 3.** `LibrarySettings` is consumed by Steps 5+ (reranker strategy, planning_mode, confidence_threshold). Implement utils immediately after schemas so all subsequent steps can import from config.

**Files:**

- `src/agents/librarian/utils/config.py` — `LibrarySettings` (pydantic-settings)
- `src/agents/librarian/utils/logging.py` — `configure_logging`, `get_logger` (structlog)
- `src/agents/librarian/utils/tracing.py` — `get_langfuse_handler`

### `LibrarySettings`

```python
class LibrarySettings(BaseSettings):
    anthropic_api_key: str = ""
    anthropic_model_haiku: str = "claude-haiku-4-5-20251001"
    anthropic_model_sonnet: str = "claude-sonnet-4-6"

    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "library-chunks"
    opensearch_user: str = "admin"
    opensearch_password: str = ""

    embedding_model: str = "intfloat/multilingual-e5-large"  # swap to "intfloat/e5-large-v2" for English-only
    ingestion_strategy: str = "html_aware"
    retrieval_strategy: str = "opensearch"
    reranker_strategy: str = "cross_encoder"
    planning_mode: Literal["rule_based", "llm"] = "rule_based"

    confidence_threshold: float = 0.4
    retrieval_k: int = 10
    reranker_top_k: int = 3
    max_query_variants: int = 3
    max_crag_retries: int = 1
    confirm_expensive_ops: bool = False   # cost gate for generate_synthetic + answer_eval

    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```

**Embedding model options** (configured via `EMBEDDING_MODEL` env var):

- `intfloat/multilingual-e5-large` — default; 100+ languages, 1024-dim, ~560MB
- `intfloat/e5-large-v2` — English-only, same dim, ~20% faster
- `intfloat/e5-small-v2` — English-only, 384-dim, lightweight for low-resource environments

### `tracing.py`

```python
def get_langfuse_handler(session_id: str, user_id: str | None = None) -> CallbackHandler | None:
    """Returns LangFuse CallbackHandler if LANGFUSE_ENABLED=true, else None."""
```

No tests for utils (config validation + logger are covered by usage in other tests).

---

## Step 3 — Ingestion module

**Files:**

- `src/agents/librarian/ingestion/base.py` — `Chunker` Protocol + `ChunkerConfig`
- `src/agents/librarian/ingestion/html_aware.py` — `HtmlAwareChunker`
- `src/agents/librarian/ingestion/parent_doc.py` — `ParentDocChunker`
- `tests/librarian/unit/test_ingestion.py`

### `base.py`

```python
class ChunkerConfig(BaseModel):
    max_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 50

class Chunker(Protocol):
    def chunk_document(self, doc: dict) -> list[Chunk]: ...
```

### `HtmlAwareChunker`

- Detect heading boundaries via regex (h1–h3 markers in plain text)
- Per-section recursive split: `\n\n → \n → sentence → word`
- Handles both `doc["text"]` and `doc["full_text"]` (field variance across source connectors)
- Drop chunks below `min_tokens`
- `_make_doc_id(url, section)` → SHA256 16-char hash

### `ParentDocChunker`

- Small child chunks (indexed for retrieval) tagged with `parent_id`
- Parent = full section chunk (returned at generation time)
- Child chunk IDs: `{parent_id}_child{i}`

**Test:** multi-sentence docs, short section bodies (< min_tokens dropped), both field names,
parent/child ID linking.

---

## Step 4 — Retrieval module

**Files:**

- `src/agents/librarian/retrieval/base.py` — `Retriever` Protocol, `Embedder` Protocol
- `src/agents/librarian/retrieval/opensearch.py` — `OpenSearchRetriever`
- `src/agents/librarian/retrieval/inmemory.py` — `InMemoryRetriever` (for tests)
- `src/agents/librarian/retrieval/embedder.py` — `MultilingualEmbedder` (E5)
- `src/agents/librarian/retrieval/mock_embedder.py` — `MockEmbedder`
- `tests/librarian/unit/test_retrieval.py`

### `base.py`

```python
class Embedder(Protocol):
    # E5 prefix rule lives here — enforced at protocol level
    def embed_query(self, text: str) -> list[float]: ...      # adds "query: "
    def embed_passage(self, text: str) -> list[float]: ...    # adds "passage: "
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

class Retriever(Protocol):
    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]: ...

    async def upsert(self, chunks: list[Chunk]) -> None: ...
```

### `OpenSearchRetriever`

- Async OpenSearch client
- `hybrid_search`: BM25 weight + k-NN weight configurable (default 0.3/0.7)
- Bulk upsert with embedding field
- BM25 analyzer configured via `OPENSEARCH_BM25_LANGUAGE` env var (default `"english"`) —
  set to the corpus language; wrong analyzer silently degrades recall on non-English text

### `InMemoryRetriever`

- Stores chunks in a list
- Cosine similarity for vector search
- BM25-like term overlap for keyword search
- Combines with configurable weights (mirrors OpenSearch behavior for tests)
- No Docker dependency — all unit tests use this

### `MultilingualEmbedder`

- SentenceTransformer wrapper
- `embed_query` adds `"query: "` prefix (E5 requirement)
- `embed_passage` adds `"passage: "` prefix
- Model from config: `EMBEDDING_MODEL=intfloat/multilingual-e5-large`
- Swap to `e5-large-v2` for English-only corpora (same dim, faster)

### `MockEmbedder`

- Returns random fixed-dimension vectors (default 1024-dim matching E5)
- Seed-stable: `np.random.default_rng(seed=42).random(dim)`
- Used in all unit tests — no model load

**Also populate `tests/librarian/unit/conftest.py` here** with the real fixtures:
`MockEmbedder`, `InMemoryRetriever`, and a mock LLM (patch `ChatAnthropic.ainvoke`).

**Test:** E5 prefix enforcement, InMemoryRetriever search correctness, upsert/search round-trip.

---

## Step 5 — Reranker module

**Files:**

- `src/agents/librarian/reranker/base.py` — `Reranker` Protocol
- `src/agents/librarian/reranker/cross_encoder.py` — `CrossEncoderReranker`
- `src/agents/librarian/reranker/llm_listwise.py` — `LLMListwiseReranker`
- `tests/librarian/unit/test_reranker.py`

### `base.py`

```python
class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        chunks: list[GradedChunk],
        top_k: int = 3,
    ) -> list[RankedChunk]: ...
    # Returns list sorted by relevance_score desc, len <= top_k
    # Sets RankedChunk.relevance_score ∈ [0, 1] and .rank (1-indexed)
```

### `CrossEncoderReranker`

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (via sentence-transformers)
- Loaded once at instantiation (not per-request)
- Scores each (query, chunk.text) pair
- Applies sigmoid to logit → [0, 1] relevance score
- Returns top-k by score

### `LLMListwiseReranker`

- Structured output call to Haiku:
  ```
  Given the query and these N documents, rank them by relevance.
  Return a JSON list: [{rank: 1, doc_index: 3, relevance_score: 0.95}, ...]
  ```
- Parses response into `RankedChunk` list
- Fallback on total parse failure: return input order with scores 0.5
- Partial parse fallback: missing indices (ranked fewer chunks than received) are appended
  at score 0.5 in original input order, so `confidence_score` is never computed from an
  incomplete list
- Used for experiments, not prod default

**Test:** cross-encoder with mock model (patch `CrossEncoder.predict`), LLM listwise with mock
`ainvoke`, top-k filtering, score range.

---

## Step 6 — Generation module

**Files:**

- `src/agents/librarian/generation/prompts.py` — system prompt constants
- `src/agents/librarian/generation/generator.py` — `build_prompt`, `call_llm`, `extract_citations`
- `tests/librarian/unit/test_generator.py`

### `prompts.py`

```python
SYSTEM_PROMPTS: dict[str, str] = {
    "lookup": "You are a precise research assistant. Answer directly from the provided sources...",
    "explore": "You are a research assistant. Synthesize findings across the provided sources...",
    "compare": "You are a research assistant. Compare the options clearly and concisely...",
    "conversational": "You are a friendly assistant. Respond naturally...",
    "out_of_scope": "You are a research assistant. This question is outside the available corpus...",
}
```

### `generator.py`

```python
def build_prompt(
    state: LibrarianState,
    ranked_chunks: list[RankedChunk],
) -> tuple[str, list[dict]]:
    """Returns (system_prompt, messages_for_llm).
    Direct intents (conversational, out_of_scope): no context injected.
    Retrieval intents: context = top-k ranked chunks joined with ---
    """

async def call_llm(
    llm: ChatAnthropic,
    system: str,
    messages: list[dict],
) -> str: ...

def extract_citations(ranked_chunks: list[RankedChunk]) -> list[dict]:
    """Returns [{"url": ..., "title": ...}] deduplicated by URL."""
```

**Test:** prompt assembly per intent, context injection, citation extraction, mock LLM call.

---

## Step 7 — Query understanding

**Files:**

- `src/agents/librarian/orchestration/query_understanding.py` — `QueryAnalyzer`, `QueryRouter`
- `tests/librarian/unit/test_query_understanding.py`

### `QueryAnalyzer`

- Rule-based keyword intent classification
- Query expansion (TERM_EXPANSIONS dictionary — domain-agnostic defaults)
- Entity extraction (regex: identifiers, dates, quantities, version numbers)
- Sub-query decomposition
- Complexity scoring (`simple / moderate / complex`)
- Returns `QueryAnalysis(intent, confidence, entities, sub_queries, complexity, expanded_terms)`

**Key rule:** `QueryAnalyzer.analyze()` must return `Intent` enum values directly — not raw
strings. Never add an INTENT_MAP translation layer in the calling graph node.

### `QueryRouter`

```python
class QueryRouter:
    def route(self, analysis: QueryAnalysis) -> Literal["retrieve", "direct", "clarify"]:
        # "direct" → conversational, out_of_scope (skip retrieval)
        # "clarify" → confidence < threshold
        # "retrieve" → all other intents
```

**Test:** each intent routes correctly, confidence threshold for clarify, edge cases.

---

## Step 8 — Retrieval subgraph

**File:** `src/agents/librarian/orchestration/subgraphs/retrieval.py`
**Test:** `tests/librarian/unit/test_subgraph_retrieval.py`

```
RetrievalState (extends LibrarianState fields relevant to retrieval)
  Must include: messages, query, standalone_query, query_variants,
                retrieved_chunks, graded_chunks, retry_count
  messages is required: rewrite_query reads conversation history to detect coreference.

Nodes:
  rewrite_query    ← Haiku call if multi-turn + coreference signals (else pass-through)
  expand_queries   ← QueryAnalyzer.sub_queries → query_variants (rule-based, no LLM)
  retrieve         ← parallel retrieval for each variant, deduplicate by chunk.id
  grade_docs       ← Haiku judges each chunk relevance (score ≥ 0.5 = relevant)
  check_sufficient ← if relevant chunks ≥ 1 → sufficient; else retry or stop

Edges:
  START → rewrite_query → expand_queries → retrieve → grade_docs → check_sufficient
  check_sufficient → [sufficient] → END
  check_sufficient → [retry] → rewrite_query  (when retry_count < 2)
  check_sufficient → [stop]  → END            (when retry_count >= 2)
```

**CRAG gotcha:** `grade_docs` increments `retry_count` on entry (before returning), so
`check_sufficient` sees count=1 after first run. Allow retry when `retry_count < 2` = exactly
one actual retry. Document this explicitly with a comment.

**Test:** sufficient path, CRAG retry fires once then stops, rewrite skipped for single-turn,
parallel retrieval with duplicate dedup.

---

## Step 9 — Reranker subgraph

**File:** `src/agents/librarian/orchestration/subgraphs/reranker.py`
**Test:** `tests/librarian/unit/test_subgraph_reranker.py`

```
Nodes:
  rerank       ← calls Reranker.rerank(query, graded_chunks, top_k=3)
  set_confidence ← confidence_score = max(chunk.relevance_score for chunk in reranked_chunks)

Edges:
  START → rerank → set_confidence → END
```

**Note:** if `graded_chunks` is empty (all CRAG graded irrelevant, retry exhausted), reranker
receives empty list → returns empty `reranked_chunks`, `confidence_score = 0.0`.

**Test:** normal rerank, empty input → empty output + zero confidence, top_k trimming.

---

## Step 10 — Generation subgraph

**File:** `src/agents/librarian/orchestration/subgraphs/generation.py`
**Test:** `tests/librarian/unit/test_subgraph_generation.py`

```
Nodes:
  build_prompt     ← assemble system prompt + context from reranked_chunks
  call_llm         ← Claude Sonnet ainvoke
  extract_citations ← deduplicate URLs from reranked_chunks metadata

Edges:
  START → build_prompt → call_llm → extract_citations → END
```

**Direct path (conversational/out_of_scope):** `skip_retrieval=True` → `build_prompt` injects
no context, uses conversational/out_of_scope system prompt.

**Test:** prompt builds correctly per intent, citations extracted and deduped, mock LLM call.

---

## Step 11 — Registry

**File:** `src/agents/librarian/registry.py`
**Test:** `tests/librarian/unit/test_registry.py`

```python
# Explicit registration — no decorator side effects, no import-order magic
from agents.library.ingestion.html_aware import HtmlAwareChunker
from agents.library.ingestion.parent_doc import ParentDocChunker
from agents.library.retrieval.opensearch import OpenSearchRetriever
from agents.library.retrieval.inmemory import InMemoryRetriever
from agents.library.reranker.cross_encoder import CrossEncoderReranker
from agents.library.reranker.llm_listwise import LLMListwiseReranker

Registry.register("chunker", "html_aware", HtmlAwareChunker)
Registry.register("chunker", "parent_doc", ParentDocChunker)
Registry.register("retriever", "opensearch", OpenSearchRetriever)
Registry.register("retriever", "inmemory", InMemoryRetriever)
Registry.register("reranker", "cross_encoder", CrossEncoderReranker)
Registry.register("reranker", "llm_listwise", LLMListwiseReranker)
```

`Registry.clear()` lives only in `conftest.py` — never in application code. Use an autouse
fixture so every test starts with a clean registry:

```python
@pytest.fixture(autouse=True)
def reset_registry():
    Registry.clear()
    yield
    Registry.clear()
```

This fixture is defined in `tests/librarian/unit/conftest.py` (scaffolded in Step 1, wired here).

**Test:** create by name, list registered strategies, unknown name raises `KeyError` with
helpful message. (Isolation is guaranteed by autouse fixture — no per-test clear needed.)

---

## Step 12 — Supervisor graph

**File:** `src/agents/librarian/orchestration/supervisor.py`
**Test:** `tests/librarian/unit/test_supervisor.py`

```python
class LibrarianGraph:
    """Entry point. Wires subgraphs into a compiled LangGraph supervisor."""

    def __init__(
        self,
        retriever: Retriever,
        embedder: Embedder,
        reranker: Reranker,
        llm: ChatAnthropic,
        checkpointer: BaseCheckpointSaver | None = None,  # MemorySaver default
        confidence_threshold: float = 0.4,
    ): ...

    def build(self) -> CompiledStateGraph: ...
```

**Graph structure:**

```
START → plan_node
  → [direct]  → generation_subgraph → confidence_gate → END
  → [clarify] → clarify_node → END
  → [retrieve] → retrieval_subgraph → reranker_subgraph → generation_subgraph → confidence_gate → END
```

**`plan_node`** (supervisor-level, not a subgraph):

- Calls `QueryAnalyzer.analyze()` + `QueryRouter.route()`
- `QueryRouter.route()` returns `"retrieve" | "direct" | "clarify"` — these must be mapped
  to node names before passing to `Command(goto=...)`:
  ```python
  _ROUTE_MAP = {
      "retrieve": "retrieval_subgraph",
      "direct": "generation_subgraph",
      "clarify": "clarify_node",
  }
  ```
- Returns `Command(goto=_ROUTE_MAP[route], update={intent, plan, skip_retrieval})`

**`confidence_gate`** (supervisor-level node):

- `confident = state["confidence_score"] >= threshold`
- If not confident: replace `response` with a "no confident answer found" message,
  set `fallback_requested = True`
- Simple rule, no LLM call

**Routing via `Command`:**

```python
def plan_node(state: LibrarianState) -> Command:
    analysis = _analyzer.analyze(state["query"])
    route = _router.route(analysis)
    return Command(
        goto=_ROUTE_MAP[route],
        update={"intent": analysis.intent.value, "plan": plan, "skip_retrieval": route != "retrieve"},
    )
```

**Factory function:**

```python
def build_library_graph(
    *,
    retriever: Retriever | None = None,   # defaults to InMemoryRetriever for tests
    embedder: Embedder | None = None,     # defaults to MockEmbedder for tests
    reranker: Reranker | None = None,     # defaults to CrossEncoderReranker
    llm: ChatAnthropic | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    confidence_threshold: float = 0.4,
) -> CompiledStateGraph: ...
```

**Test:** full pipeline happy path, conversational skips retrieval, low confidence sets
`fallback_requested`, CRAG retry fires + stops, mock all LLM calls and retriever.

---

## Step 13 — Utils

**Moved to Step 2.5.** Utils must be implemented before ingestion — see Step 2.5 above.

---

## Step 14 — Eval suite

**Files:**

- `src/agents/librarian/evals/models.py`
- `src/agents/librarian/evals/extract_golden.py`
- `src/agents/librarian/evals/retrieval_eval.py`
- `src/agents/librarian/evals/answer_eval.py`
- `src/agents/librarian/evals/generate_synthetic.py`
- `tests/librarian/unit/test_evals.py`

### `models.py`

```python
class GoldenSample(BaseModel):
    query_id: str
    query: str
    expected_doc_url: str
    relevant_chunk_ids: list[str]
    category: str
    language: str = "en"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    validation_level: Literal["gold", "silver", "bronze"] = "silver"
    source_record_id: str | None = None   # upstream record from ingestion source

class RetrievalMetrics(BaseModel):
    hit_rate_at_k: float
    mrr: float
    k: int
    n_queries: int

class EvalRunConfig(BaseModel):
    prompt_version: str
    model_id: str
    corpus_version: str
    reranker_strategy: str
    top_k: int
    notes: str = ""
    timestamp: str   # ISO 8601
```

### `extract_golden.py`

- Tiered extraction: gold (hand-curated with chunk IDs) / silver (human-validated) / bronze (inferred from interaction logs)
- Deduplication keyed on `(query, doc_url)` pair — NOT record ID
- CLI: `python -m agents.library.evals.extract_golden --records data/records.jsonl --tier silver`

### `retrieval_eval.py`

- `evaluate_retrieval(golden: list[GoldenSample], retrieve_fn, k=5) -> RetrievalMetrics`
- hit_rate@k: % of queries where expected URL appears in top-k results
- MRR: mean reciprocal rank of the first relevant result
- Failure clustering: group failed queries by pattern (no results / wrong intent / score too low)

### `answer_eval.py`

- `AnswerJudge`: Haiku-based evaluator
- Evaluates: faithfulness, relevance, completeness (0–1 scores)
- Returns `JudgeResult(is_correct, score, reasoning)`
- deepeval integration: `@pytest.mark.deepeval` for CI regression
- Cost gate: `settings.confirm_expensive_ops` (from `LibrarySettings`) — never commit as True

### `generate_synthetic.py`

- Takes a corpus of chunks, generates (query, expected_doc_url) pairs via LLM
- Output: JSONL compatible with `GoldenSample` schema
- Cost gate: `settings.confirm_expensive_ops` (from `LibrarySettings`) — never commit as True

**Test:** metrics computation (hit@k, MRR), golden extraction tiers + dedup, judge scoring
with mocked LLM.

---

## File map — final layout

```
src/agents/librarian/
  __init__.py
  registry.py
  schemas/
    __init__.py
    chunks.py          # Chunk, ChunkMetadata, GradedChunk, RankedChunk
    retrieval.py       # Intent, RetrievalResult, QueryPlan
    state.py           # LibrarianState
  ingestion/
    __init__.py
    base.py            # Chunker Protocol, ChunkerConfig
    html_aware.py      # HtmlAwareChunker
    parent_doc.py      # ParentDocChunker
  retrieval/
    __init__.py
    base.py            # Retriever Protocol, Embedder Protocol
    opensearch.py      # OpenSearchRetriever
    inmemory.py        # InMemoryRetriever
    embedder.py        # MultilingualEmbedder (E5, swappable via config)
    mock_embedder.py   # MockEmbedder
  reranker/
    __init__.py
    base.py            # Reranker Protocol
    cross_encoder.py   # CrossEncoderReranker
    llm_listwise.py    # LLMListwiseReranker
  generation/
    __init__.py
    prompts.py         # SYSTEM_PROMPTS dict
    generator.py       # build_prompt, call_llm, extract_citations
  orchestration/
    __init__.py
    query_understanding.py  # QueryAnalyzer, QueryRouter
    supervisor.py           # LibrarianGraph, build_library_graph()
    subgraphs/
      __init__.py
      retrieval.py     # retrieval_subgraph (rewrite → expand → retrieve → grade → check)
      reranker.py      # reranker_subgraph (rerank → set_confidence)
      generation.py    # generation_subgraph (build_prompt → call_llm → citations)
  evals/
    __init__.py
    models.py          # GoldenSample, RetrievalMetrics, EvalRunConfig
    extract_golden.py  # tiered golden dataset extraction
    retrieval_eval.py  # hit@k, MRR, failure clustering
    answer_eval.py     # LLM-as-judge + deepeval
    generate_synthetic.py
  utils/
    __init__.py
    config.py          # LibrarySettings (pydantic-settings)
    logging.py         # structlog configure + get_logger
    tracing.py         # get_langfuse_handler (optional)

tests/librarian/
  __init__.py
  unit/
    __init__.py
    conftest.py        # shared fixtures (MockEmbedder, InMemoryRetriever, mock LLM, reset_registry)
    test_schemas.py
    test_ingestion.py
    test_retrieval.py
    test_reranker.py
    test_generator.py
    test_query_understanding.py
    test_subgraph_retrieval.py
    test_subgraph_reranker.py
    test_subgraph_generation.py
    test_registry.py
    test_supervisor.py
    test_evals.py
```

---

## Execution order

```
Step 1   — scaffold (no code; conftest stub; confirm pyproject.toml changes)
Step 2   — schemas
Step 2.5 — utils (LibrarySettings + logging + tracing — must precede Step 3+)
Step 3   — ingestion
Step 4   — retrieval (also populates conftest fixtures: MockEmbedder, InMemoryRetriever, mock LLM)
Step 5   — reranker
Step 6   — generation
Step 7   — query understanding
Step 8   → 10 — subgraphs (in order; each builds on schemas + modules)
Step 11  — registry (after all strategies exist)
Step 12  — supervisor (after all subgraphs + registry)
Step 13  — utils (moved to Step 2.5 — no-op here)
Step 14  — evals (after schemas + retrieval)
```

All tests must pass without Docker, API keys, or model downloads.
MockEmbedder + InMemoryRetriever + mocked LLM calls cover everything.

---

## Risks and gotchas

1. **CRAG retry counter off-by-one**: `grade_docs` increments `retry_count` on entry. `check_sufficient`
   allows retry when `retry_count < 2`. After first run, count = 1 → retry fires. After second run,
   count = 2 → stop. Document with inline comment; test explicitly.

2. **E5 prefix enforcement**: `embed_query` must add `"query: "`, `embed_passage` must add `"passage: "`.
   Define this in the `Embedder` Protocol docstring so implementations can't silently skip it.
   MockEmbedder does NOT need to add prefixes (returns random vectors anyway).

3. **Cross-encoder model at startup**: `CrossEncoderReranker` loads the model in `__init__`.
   In tests, patch `sentence_transformers.CrossEncoder` to avoid download.

4. **Intent enum values directly from QueryAnalyzer**: `QueryAnalyzer.analyze()` must return
   `Intent` enum values — not raw strings. Never add an INTENT_MAP translation in the calling node.

5. **`total=False` on `LibrarianState`**: nodes return partial dicts; LangGraph merges them.
   Tests must handle missing fields with `.get("field", default)` — never direct key access.

6. **`Command` routing in LangGraph**: supervisor routes to subgraphs via `Command(goto=..., update=...)`.
   Subgraphs must be added to the supervisor graph as nodes using `add_node("name", subgraph)`.
   Subgraph state is merged into supervisor state on return.

7. **LangFuse optional**: `get_langfuse_handler` returns `None` when disabled. LangGraph
   `config["callbacks"]` must handle `None` — pass as `[handler] if handler else []`.

8. **BM25 analyzer language**: `OpenSearchRetriever` BM25 must use the right language analyzer
   for the corpus. Configure via `OPENSEARCH_BM25_LANGUAGE` — default `"english"`. Using the
   wrong analyzer silently degrades recall on non-English text (no error raised).
   `InMemoryRetriever` is language-agnostic (term overlap, no stemming).

9. **Reranker at small corpus**: cross-encoder reranking can degrade quality at <1K chunks
   (overfits to surface similarity). Configure `RERANKER_MIN_CORPUS_SIZE` below which
   reranker is skipped.

10. **Response style matters as much as retrieval**: correct content that is 3× too long or
    uses hedging language is ignored by users. Generation prompts must enforce direct,
    actionable language. Confidence gate should set `fallback_requested` and return a clean
    "no confident answer" — not a hedged response.

11. **`messages` is append-only; all other LibrarianState fields replace**: nodes returning
    `{"messages": [...]}` append via the `add_messages` reducer. All other fields are
    last-writer-wins. Never assert `state["messages"][0]` in tests — use `state["messages"][-1]`
    or check the full list.

12. **`plan_node` route strings don't match node names**: `QueryRouter.route()` returns
    `"retrieve" | "direct" | "clarify"`. `Command(goto=...)` needs the node name
    (`"retrieval_subgraph" | "generation_subgraph" | "clarify_node"`). Missing the `_ROUTE_MAP`
    is a runtime `KeyError` on first invocation — not caught at compile time.
