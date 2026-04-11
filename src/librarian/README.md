# Librarian — RAG Agent Template

A lean, observable RAG system grounded in observability, feedback, and evaluation — forming the foundation for self-learning, agentic intelligence.

> Every conversation contributes to system improvement. Failures are first-class signals with diagnostic precision, making continuous learning tractable. Not just a chatbot, but a librarian that dynamically maps the knowledge system to the needs and context of each user.

---

## Why this exists

Domain knowledge is fragmented, fast-changing, and context-dependent, while user expectations are instant and high-precision. Traditional retrieval systems — keyword search, static documents, naive chatbots — fail to reliably map user intent to the right information in real time.

Current AI systems introduce a new failure mode: they can produce confident but incorrect answers. These failures are difficult to detect, diagnose, and improve without strong observability.

**Answer quality** is defined across five dimensions:
- **Relevance** — does the answer directly address the user's need?
- **Reliability** — is it factually grounded and consistent with source knowledge?
- **Control & transparency** — can we trace how the answer was produced and why?
- **Accessibility & speed** — is it easy to understand and delivered instantly?
- **Depth vs efficiency** — does it balance completeness with brevity based on user intent?

**North star metric**: query resolution rate — share of queries fully resolved without fallback or human intervention.

Supporting signals: hallucination rate, friction detection rate (rephrasing, follow-ups, abandonment, escalation), eval harness pass rate.

---

## Architecture

The system is structured as four agents orchestrated by a LangGraph state machine:

```
Query
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Orchestrator (Planning Agent)                              │
│  Intent classification → retrieval plan → routing decision  │
└─────────────┬───────────────────────────────────────────────┘
              │
      ┌───────┴────────┐
      ▼                ▼
 [direct]         [retrieve]
      │                │
      │     ┌──────────▼──────────┐
      │     │  Retrieval Agent    │
      │     │  Embed + Hybrid     │
      │     │  Search + Snippets  │
      │     └──────────┬──────────┘
      │                │
      │     ┌──────────▼──────────┐
      │     │  Re-Ranking Agent   │
      │     │  Cross-encoder +    │
      │     │  confidence score   │
      │     └──────────┬──────────┘
      │                │
      │          [confidence gate]
      │          low? ──► CRAG retry ──► Retrieval Agent
      │          ok?  ──► continue
      │                │
      └────────────────┤
                       ▼
             ┌─────────────────┐
             │ Generation Agent│
             │ LLM + Prompt    │
             │ Library + Cite  │
             └────────┬────────┘
                      │
                   Response + Citations
```

### Agent responsibilities

**Planning Agent** — Query understanding + orchestration
- Intent classification (lookup / explore / compare / conversational / out_of_scope)
- Clarification detection — decides when to ask for more information
- Context augmentation — determines what additional user/system context is needed
- Plan generation — defines retrieval strategy and downstream path
- Confidence-aware routing — sets expectations for downstream thresholds

**Retrieval Agent** — Knowledge access and recall
- Bi-encoder semantic retrieval (E5 multilingual embeddings)
- Hybrid search (semantic + BM25 keyword)
- Metadata filtering
- Multi-query expansion (up to N variants, deduplicated by chunk ID)
- Recall optimization — broad coverage at this stage

**Re-Ranking Agent** — Precision layer
- Cross-encoder or LLM-based re-ranking
- Fine-grained query-document relevance scoring
- Context deduplication and diversity enforcement
- Confidence scoring on final context set
- Insufficient context detection — signals when results aren't good enough

**Generation Agent** — Answer synthesis
- Prompt orchestration (intent-aware templates, dynamic assembly)
- Grounded response generation — strictly tied to retrieved context
- Source attribution and citation
- Clarification generation when triggered downstream
- Uncertainty handling when context is weak or incomplete

---

## Pipeline flow

```
Raw docs → Chunking → Embedding → Indexing → VectorDB
                                                  │
Query ──► Plan ──► Retrieve ──► Rerank ──► Gate ──► Generate ──► Response
                     ▲                       │
                     └──── CRAG retry ───────┘ (if confidence < threshold)
```

### CRAG (Corrective RAG)

When the reranker confidence score falls below `confidence_threshold`, the graph retries retrieval with a reformulated query instead of generating a low-confidence answer. The gate node increments `retry_count`; after `max_crag_retries` attempts the system generates from whatever context is available.

This catches **search failures** (wrong chunks retrieved) rather than **coverage gaps** (topic not in corpus) — the two most common hallucination root causes.

---

## Key design decisions

### Hallucination is a system-level failure

Insufficient context, poor ranking, incorrect routing, or lack of user/product context can each independently cause hallucination. The system attributes root cause across the pipeline via:
- `confidence_score` from the reranker (retrieval/ranking signal)
- `failure_reason` in trace records (`expected_doc_not_in_top_k`, `zero_retrieval`, etc.)
- `FailureClusterer` groups failures by type for batch diagnosis

### Observability is not optional

Failures are hard to localize (planning vs retrieval vs generation) without structured traces. Every node emits structured log events (`graph.crag.retry`, `chroma.upsert.done`, etc.) via structlog. LangFuse integration is opt-in for trace-level debugging and metric dashboards.

### User satisfaction is inferred, not measured

Explicit feedback (thumbs up/down) is sparse, delayed, and biased. The system infers quality from **friction signals**: rephrasing, follow-ups, abandonment, escalation. These feed the eval dataset and drive the feedback loop.

---

## Component choices

| Stage | Selected | Local/Free | Swap path |
|---|---|---|---|
| Chunking | `HtmlAwareChunker` (heading-boundary recursive) | ✓ | `ParentDocChunker` for long docs |
| Embeddings | `intfloat/multilingual-e5-large` (1024-dim) | ✓ | `e5-large-v2` (English-only), Voyage (cloud) |
| Vector store | ChromaDB (persistent, HNSW) | ✓ | OpenSearch (prod), Qdrant (scale) |
| Hybrid retrieval | Term-overlap + cosine (0.3/0.7) | ✓ | Native BM25+kNN in OpenSearch |
| Reranking | `ms-marco-MiniLM-L-6-v2` cross-encoder | ✓ | `bge-reranker-large`, Cohere API |
| Generation | Claude Sonnet 4.6 (LangChain) | API key | Haiku (cheap), Ollama (free) |
| Query planner | Rule-based intent classifier | ✓ | LLM classifier (`planning_mode=llm`) |
| Orchestration | LangGraph CRAG state machine | ✓ | Custom pipeline |
| Observability | structlog + LangFuse (opt-in) | ✓ | LangSmith, OpenTelemetry |

Full rationale and alternatives in [`.claude/docs/plans/research/rag-tradeoffs.md`](../../.claude/docs/plans/research/rag-tradeoffs.md).

---

## Eval suite

The eval suite mirrors the V2 architecture from the design: Tasks → Trials → Graders → Metrics, integrated with pytest/CI.

```
tests/librarian/evalsuite/
  conftest.py                      # golden dataset (5 samples), shared fixtures
  regression/
    test_retrieval_metrics.py      # hit_rate@5 ≥ 0.6, MRR ≥ 0.4 — never lower these
  capability/
    test_pipeline_capability.py    # end-to-end routing, CRAG termination, state keys
    test_answer_judge.py           # AnswerJudge grader, cost gate, parse/API error handling
```

**Graders** (in `eval_harness/graders/`):
- `AnswerJudge` — LLM-based: scores faithfulness, relevance, completeness (guarded by `CONFIRM_EXPENSIVE_OPS`)
- Code-based: hit_rate@k, MRR, chunk ID matching

**Metrics** (in `eval_harness/metrics/`):
- `evaluate_retrieval()` → `(RetrievalMetrics, list[FailureCluster])`
- `FailureClusterer` groups misses by type (`zero_retrieval`, `expected_doc_not_in_top_k`, etc.)
- LangFuse score push (opt-in via `langfuse_trace_id`)

**Feedback loop**: conversation traces + friction signals (escalation, abandonment, rephrasing) → eval dataset → regression and capability tests → CI gate.

---

## Setup

```bash
# Install with librarian extras
uv sync --extra librarian
uv add chromadb  # local vector store

# Required
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env

# Run all tests (unit + evalsuite)
uv run pytest tests/librarian/

# Retrieval regression only (fast, no LLM calls)
uv run pytest tests/librarian/evalsuite/regression/

# Unit tests only
uv run pytest tests/librarian/unit/
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | Claude API key (personal account is fine) |
| `ANTHROPIC_MODEL_SONNET` | `claude-sonnet-4-6` | Generation model |
| `ANTHROPIC_MODEL_HAIKU` | `claude-haiku-4-5-20251001` | Cheap routing / classification |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | SentenceTransformer model |
| `RETRIEVAL_STRATEGY` | `chroma` | `chroma` / `opensearch` / `inmemory` |
| `CHROMA_PERSIST_DIR` | `.chroma` | Local Chroma persistence path |
| `CHROMA_COLLECTION` | `librarian-chunks` | Collection name |
| `CONFIDENCE_THRESHOLD` | `0.4` | CRAG gate — retry below this score |
| `MAX_CRAG_RETRIES` | `1` | Max CRAG loop iterations |
| `RETRIEVAL_K` | `10` | Candidates fetched from vector store |
| `RERANKER_TOP_K` | `3` | Chunks passed to generation |
| `LANGFUSE_ENABLED` | `false` | Enable LangFuse tracing |

---

## Project structure

```
src/agents/librarian/
  factory.py                    # create_librarian() — injectable entry point
  schemas/
    state.py                    # LibrarianState TypedDict
    chunks.py                   # Chunk, GradedChunk, RankedChunk
    retrieval.py                # RetrievalResult, QueryPlan
  orchestration/
    graph.py                    # LangGraph StateGraph (build_graph)
    query_understanding.py      # QueryAnalyzer, QueryRouter
    subgraphs/
      retrieval.py              # RetrievalSubgraph (multi-query, grade, dedup)
      reranker.py               # RerankerSubgraph (filter, rerank, confidence)
      generation.py             # GenerationSubgraph (prompt, LLM, cite, gate)
  retrieval/
    base.py                     # Embedder + Retriever Protocols
    embedder.py                 # MultilingualEmbedder (E5 prefix rule)
    chroma.py                   # ChromaRetriever (default, local)
    opensearch.py               # OpenSearchRetriever (production)
    inmemory.py                 # InMemoryRetriever (tests only)
    mock_embedder.py            # MockEmbedder (deterministic, tests only)
  reranker/
    base.py                     # Reranker Protocol
    cross_encoder.py            # CrossEncoderReranker (default)
    llm_listwise.py             # LLMListwiseReranker (high-value queries)
  generation/
    generator.py                # GenerationNode (LLM call, citation extraction)
    prompts.py                  # Intent-aware system prompt library
  ingestion/
    base.py                     # ChunkerConfig
    html_aware.py               # HtmlAwareChunker (heading-boundary recursive)
    parent_doc.py               # ParentDocChunker (stub, future)
  eval_harness/
    graders/answer_eval.py      # AnswerJudge, ClosedBookBaseline, JudgeResult
    metrics/retrieval_eval.py   # evaluate_retrieval(), FailureClusterer
    tasks/models.py             # GoldenSample, RetrievalMetrics
    tasks/tracing.py            # PipelineTracer, FailureCluster
  utils/
    config.py                   # LibrarySettings (pydantic-settings)
    logging.py                  # structlog get_logger
    registry.py                 # Component registry
    tracing.py                  # Trace helpers
```
