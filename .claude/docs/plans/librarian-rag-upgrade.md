# Plan: Librarian RAG Upgrade — Multi-Query, Structured Output, Tool Abstraction

**Status:** Draft — awaiting review  
**Scope:** `src/librarian/`, `src/orchestration/` — RAG-only (no agent autonomy expansion)  
**Depends on:** [langgraph-adk-compatibility.md](langgraph-adk-compatibility.md) (partially overlapping — this plan supersedes the ADK tools rewrite in that doc)

---

## Problem Statement

The Librarian pipeline works but has gaps compared to production-grade RAG patterns:

1. **Multi-query retrieval is ad-hoc.** `RetrieverAgent` already supports `query_variants` via `QueryPlan`, but there's no reusable `EnsembleRetriever` primitive. Dedup is chunk-ID based (`_grade_chunks`) — no global fingerprint dedup across sources. The ADK `search_knowledge_base` tool is single-query only.

2. **Generator output is unstructured.** `call_llm()` returns raw `str`. There's no Pydantic model enforcing response shape (answer text, citations, confidence, follow-up suggestions). No use of Claude structured JSON mode.

3. **No framework-agnostic tool abstraction.** The retriever is bound to LangGraph state (`LibrarianState`) in `RetrieverAgent` and to raw protocol calls in ADK `tools.py`. There's no shared callable tool object with explicit I/O schema that both LangGraph and ADK can use without reimplementation.

4. **A few hardcoded constants remain.** `_RELEVANCE_THRESHOLD = 0.1` in `retrieval.py`, `DEFAULT_CONFIDENCE_GATE = 0.3` in `generation.py`, `RRF_K = 60` in `rrf.py` — should flow from config.

---

## What We Are NOT Doing

- **Not adding agent autonomy.** Librarian stays RAG-only — no multi-hop reasoning, no tool-use loops, no memory.
- **Not changing the CRAG state machine.** The LangGraph graph structure stays the same; we're improving the primitives it calls.
- **Not adding LangChain dependencies.** The `EnsembleRetriever` is our own class, not `langchain.retrievers.EnsembleRetriever`.
- **Not changing storage backends.** Chroma, OpenSearch, DuckDB all stay as-is.

---

## Implementation Steps

### Step 1 — `EnsembleRetriever` with fingerprint dedup

**Files:** new `src/librarian/retrieval/ensemble.py`, edit `src/librarian/retrieval/__init__.py`  
**Tests:** new `tests/librarian/unit/test_ensemble.py`

Create a standalone `EnsembleRetriever` class that:

```python
class EnsembleRetriever:
    """Multi-query, multi-retriever fusion with fingerprint dedup."""

    def __init__(
        self,
        retrievers: list[Retriever],
        embedder: Embedder,
        *,
        score_threshold: float = 0.4,       # from config
        max_queries: int = 3,                # from config
    ) -> None: ...

    async def retrieve(
        self,
        queries: list[str],               # 1–3 query variants
        k: int = 10,
    ) -> list[GradedChunk]: ...
```

Key behaviors:
- `asyncio.gather` runs all queries × all retrievers in parallel (queries × retrievers matrix)
- Fingerprint dedup: `f"{chunk.metadata.url}|{chunk.text[:200].lower().strip()}"` → SHA-256 → keep highest-scored copy
- Score threshold filtering: drop chunks below `score_threshold` (configurable via `LibrarySettings.confidence_threshold`)
- RRF fusion across all result lists using existing `fuse_rankings()`
- Returns `list[GradedChunk]` — same interface the reranker already expects

The existing `RetrieverAgent` will be updated to delegate to `EnsembleRetriever` instead of manually managing the embed → search → dedup flow. This collapses ~60 lines in `RetrieverAgent.run()` to ~5.

### Step 2 — `RAGResponse` Pydantic model + structured output

**Files:** new model in `src/librarian/schemas/response.py`, edit `src/librarian/generation/generator.py`  
**Tests:** edit `tests/librarian/unit/test_generator.py`

Define a structured response model:

```python
class Citation(BaseModel):
    url: str
    title: str
    snippet: str = ""       # relevant excerpt from source

class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]
    follow_up: str = ""     # suggested follow-up question (optional)
```

Update `call_llm()` → `call_llm_structured()` path:
- When `RAGResponse` validation is desired (retrieval intents), the system prompt instructs Claude to respond in JSON matching the schema
- Parse response with `RAGResponse.model_validate_json()` 
- Fallback: if JSON parsing fails, wrap raw text in `RAGResponse(answer=raw_text, citations=..., confidence="low")`
- `extract_citations()` is replaced by the model's `citations` field
- The `GeneratorAgent.run()` return dict gains `"response_model": RAGResponse` alongside the existing `"response": str` for backward compat

Keep the unstructured path for `conversational` and `out_of_scope` intents — those don't need JSON mode.

### Step 3 — Tool abstraction layer (`BaseTool` protocol)

**Files:** new `src/librarian/tools/base.py`, new `src/librarian/tools/retriever_tool.py`, new `src/librarian/tools/__init__.py`  
**Tests:** new `tests/librarian/unit/test_tools.py`

Define a minimal tool protocol that both LangGraph and ADK can consume:

```python
class ToolInput(BaseModel):
    """Override in subclasses to define tool input schema."""
    ...

class ToolOutput(BaseModel):
    """Override in subclasses to define tool output schema."""
    ...

@runtime_checkable
class BaseTool(Protocol):
    name: str
    description: str
    input_schema: type[ToolInput]
    output_schema: type[ToolOutput]

    async def run(self, input: ToolInput) -> ToolOutput: ...
```

Concrete `RetrieverTool`:

```python
class RetrieverToolInput(ToolInput):
    queries: list[str] = Field(min_length=1, max_length=3)
    num_results: int = Field(default=10, ge=1, le=50)

class RetrieverToolOutput(ToolOutput):
    results: list[GradedChunkDict]
    total: int
    deduplicated: int

class RetrieverTool:
    name = "search_knowledge_base"
    description = "Multi-query hybrid search over the knowledge base"
    input_schema = RetrieverToolInput
    output_schema = RetrieverToolOutput

    def __init__(self, ensemble: EnsembleRetriever) -> None: ...

    async def run(self, input: RetrieverToolInput) -> RetrieverToolOutput:
        chunks = await self.ensemble.retrieve(input.queries, k=input.num_results)
        return RetrieverToolOutput(results=..., total=..., deduplicated=...)
```

**ADK adapter** (update `src/orchestration/google-adk/tools.py`):
- `search_knowledge_base()` becomes a thin wrapper: build `RetrieverToolInput` from args → call `RetrieverTool.run()` → return `.model_dump()`
- Same pattern for `rerank_results` → `RerankerTool`

**LangGraph adapter** (update `src/orchestration/langgraph/nodes/retrieval.py`):
- `RetrieverAgent.__init__` accepts a `RetrieverTool` (or `EnsembleRetriever` directly)
- `RetrieverAgent.run()` delegates to it

This makes it trivial to wrap any tool for a new framework (LangGraph ToolNode, ADK FunctionTool, or a future Python agent SDK) without reimplementation.

### Step 4 — Config-driven constants ✅

**Files:** edit `src/librarian/config.py`, edit files referencing hardcoded values  
**Tests:** edit existing config/factory tests

Move remaining hardcoded values to `LibrarySettings`:

| Constant | Current location | Config field | Default |
|---|---|---|---|
| `_RELEVANCE_THRESHOLD = 0.1` | `retrieval.py:17` | `relevance_threshold` | **already exists** (0.1) — just wire it |
| `DEFAULT_CONFIDENCE_GATE = 0.3` | `generation.py:19` | `confidence_threshold` | **already exists** (0.4) — use it, remove local override |
| `RRF_K = 60` | `rrf.py:18` | new: `rrf_k` | 60 |
| `_CONTEXT_SEP` | `generator.py:15` | leave as-is (not tunable) | — |
| `score_threshold` in `EnsembleRetriever` | Step 1 | `confidence_threshold` | 0.4 |

Notable: `DEFAULT_CONFIDENCE_GATE` (0.3) in `generation.py` **disagrees** with `confidence_threshold` (0.4) in config. The factory already passes `cfg.confidence_threshold` to `GeneratorAgent`, so the local constant is dead code — remove it and rely on the injected value.

Wire `relevance_threshold` from config into `RetrieverAgent` constructor (currently reads from the `_RELEVANCE_THRESHOLD` module constant).

### Step 5 — CI parity (Makefile + shared conventions)

**Files:** edit `Makefile`, possibly new `pyproject.toml` test markers  
**No new code in `src/`**

Align test/lint targets with the ADK-style conventions:

- Add `make test-librarian`, `make test-adk`, `make test-all` targets (currently eval-focused Makefile)
- Add `make lint` (ruff check + format check) and `make typecheck` (pyright)
- Ensure `SCORE_THRESHOLD` and `NUM_QUERIES_MAX` are validated in a `test_config.py` confirming they're config-driven, not hardcoded
- Add a test that imports both LangGraph and ADK tool wrappers and verifies they produce equivalent outputs for the same input (parity test)

---

## Dependency Graph

```
Step 1 (EnsembleRetriever)
   │
   ├──→ Step 3 (Tool abstraction — wraps EnsembleRetriever)
   │       │
   │       └──→ Step 3b (ADK + LangGraph adapters)
   │
   └──→ Step 2 (RAGResponse — independent of Step 3)

Step 4 (Config) — can start in parallel with Step 1
Step 5 (CI) — after Steps 1-4
```

**Recommended order:** Step 4 → Step 1 → Step 2 → Step 3 → Step 5

Start with Step 4 (config cleanup) because it unblocks clean constructors for Steps 1-3. Step 2 (structured output) is independent and can run in parallel with Step 3 once Step 1 is done.

---

## Risk & Mitigations

| Risk | Mitigation |
|---|---|
| Fingerprint dedup misses near-duplicates with different URLs | Fingerprint is `url|content[:200]` — catches exact and near-duplicates from same source. Cross-source near-dupes are handled by the existing reranker. |
| Structured JSON mode fails on edge cases | Fallback wraps raw text in `RAGResponse(confidence="low")`. Test with adversarial prompts. |
| `BaseTool` protocol adds indirection | Keep it minimal (4 fields + 1 method). If it doesn't earn its keep after Step 3, collapse it. |
| Config drift between LangGraph and ADK paths | Parity test (Step 5) catches this. Both paths pull from the same `LibrarySettings` singleton. |

---

## Acceptance Criteria

- [ ] `EnsembleRetriever.retrieve(["q1", "q2"])` returns deduped, score-filtered `GradedChunks` — unit test with `MockEmbedder` + `InMemoryRetriever`
- [ ] `RAGResponse` validates generator output; fallback path tested with malformed JSON
- [ ] `RetrieverTool` has explicit `input_schema` / `output_schema`; both ADK and LangGraph use it
- [ ] Zero hardcoded threshold/tuning constants in `src/librarian/` or `src/orchestration/` — all flow from `LibrarySettings`
- [ ] `uv run pytest tests/librarian/ tests/orchestration/` passes
- [ ] Parity test confirms LangGraph and ADK produce equivalent retrieval results
