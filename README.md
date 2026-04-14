# Agents

Personal AI agent toolkit with two systems: a **Librarian RAG pipeline** for retrieval-augmented generation over technical corpora, and a set of standalone **research/presentation agents** for processing PDFs into Obsidian notes and slide decks.

## Librarian RAG System

A production-grade RAG pipeline built on LangGraph with hybrid retrieval, cross-encoder reranking, CRAG self-correction, and full observability. Designed for technical and domain-specific corpora where term overlap matters (code, product names, jargon, version numbers).

### Architecture

```
                         ┌──────────────┐
              ┌──────────│   Condense    │  (multi-turn history → standalone query)
              │          │  (Haiku LLM)  │
              │          └──────┬───────┘
              │                 ▼
              │          ┌──────────────┐
              │          │   Analyze     │  (intent classification, query expansion,
              │          │              │   retrieval mode routing)
              │          └──────┬───────┘
              │                 │
              │     ┌───────────┼────────────┐
              │     ▼           ▼            ▼
         ┌─────────┐   ┌──────────────┐  ┌──────────┐
         │ Snippet  │   │   Retrieve   │  │ Generate │  (direct — conversational/
         │ Retrieve │   │ (hybrid BM25 │  │          │   out-of-scope intents)
         │ (DuckDB  │   │  + dense +   │  └──────────┘
         │   FTS)   │   │  multi-query │
         └────┬─────┘   │    RRF)      │
              │         └──────┬───────┘
              │                ▼
              │         ┌──────────────┐
              │         │   Rerank     │  (cross-encoder: ms-marco-MiniLM-L-6-v2)
              │         └──────┬───────┘
              │                ▼
              │         ┌──────────────┐
              │         │    Gate      │  (confidence check → CRAG retry loop)
              │         └──────┬───────┘
              │                ▼
              └────────►┌──────────────┐
                        │   Generate   │  (Claude Sonnet, streaming)
                        └──────────────┘
```

**Key capabilities:**

- **Hybrid retrieval** — BM25 + dense vector search with configurable weights (default 0.3 / 0.7), fused via Reciprocal Rank Fusion (RRF)
- **Multi-query expansion** — reformulates the query into N variants, retrieves for each, merges ranked lists
- **Cross-encoder reranking** — scores each (query, chunk) pair and reorders; 10–20% hit_rate improvement over cosine-only
- **CRAG self-correction** — if confidence falls below threshold, rewrites the query and retries retrieval before generating
- **History condensation** — rewrites multi-turn queries to be self-contained (e.g. "what about the Python one?" → "What is the Python version of [topic]?")
- **3-way query routing** — dense retrieval / snippet retrieval (DuckDB FTS) / direct generation based on intent
- **Structured observability** — every node emits structured logs; confidence scores, failure reasons, and per-step timing inspectable in traces

### Three Pipeline Variants

The project defines three retrieval configurations for side-by-side comparison. All three run through the same eval harness against identical golden datasets.

| | **Librarian** | **Raptor** | **Bedrock** |
|---|---|---|---|
| **What** | Full custom RAG pipeline | Mirrors `cs_agent_assist_with_rag` params | AWS Bedrock Knowledge Bases |
| **Retrieval** | Hybrid BM25 + dense, k=10 | Pure KNN, k=5 | Pure KNN, k=5 (AWS-managed) |
| **Embeddings** | `multilingual-e5-large` (1024-dim) | `all-MiniLM-L6-v2` (384-dim) | AWS Titan v2 (1536-dim) |
| **Reranker** | CrossEncoder (`ms-marco-MiniLM`) | None (passthrough) | None |
| **BM25 weight** | 0.3 | 0.0 | 0.0 |
| **Multi-turn** | HistoryCondenser (Haiku rewrite) | N/A | AWS `sessionId` (limited) |
| **Observability** | Full per-node traces, failure clustering | Same harness (config variant) | Black box — answer + citations only |
| **Streaming** | Yes (SSE from generation node) | Same pipeline | No (blocking `RetrieveAndGenerate`) |
| **Cost** | Fixed ~$50–80/mo (Fargate) + model tokens | Same infra | Serverless per-call (~$0.001/query) |

Raptor and Bedrock are **not separate codebases** — they are named `LibrarySettings` configurations in `src/eval/variants.py` that parametrize the same retrieval code with different settings. For eval, Bedrock is approximated locally with the same KNN setup; for production, it routes through `BedrockKBClient` calling the real AWS API.

> For full analysis see [Librarian vs Bedrock KB research](.claude/docs/research/librarian-vs-bedrock-kb.md).

---

## Running the Pipeline Comparison

### Quick start — eval comparison (no API keys, no external services)

The comparison harness runs all three variants against a golden retrieval dataset using `InMemoryRetriever` + `MockEmbedder`. No API keys, no model downloads, no external services.

```bash
# Run the three-way comparison (prints side-by-side metrics table)
uv run pytest tests/librarian/evalsuite/regression/test_variant_comparison.py -v -s
```

The `-s` flag is important — it shows the printed comparison table with per-variant metrics:

```
  [librarian ]  hit_rate@10=0.800  mrr=0.650  n=5  failures=[none]
  [raptor    ]  hit_rate@5=0.600   mrr=0.400  n=5  failures=[expected_doc_not_in_top_k×2]
  [bedrock   ]  hit_rate@5=0.600   mrr=0.400  n=5  failures=[expected_doc_not_in_top_k×2]
```

**Metrics reported:**
- **hit_rate@k** — fraction of queries where the expected document appears in the top-k results
- **MRR** (Mean Reciprocal Rank) — average of 1/rank for the first relevant result
- **n** — number of golden queries evaluated
- **failures** — clustered failure types: `zero_retrieval`, `expected_doc_not_in_top_k`, `low_confidence`

### Full eval tiers

```bash
# Fast unit tests — no external deps, always safe
make eval-unit

# Retrieval metric regression — hit_rate and MRR floor assertions + variant comparison
make eval-regression

# Full capability suite — downloads ~500MB models on cold cache
# Gate: requires explicit confirmation
CONFIRM_EXPENSIVE_OPS=1 make eval-capability
```

### Running with real data

For comparison against a real corpus (external golden dataset):

```bash
# Point to your golden samples JSONL file
export EVAL_DATASET_PATH=/path/to/golden_samples.jsonl

# Run local data comparison
uv run pytest tests/librarian/evalsuite/local/test_local_data_comparison.py -v -s
```

### Live A/B comparison (Librarian vs Bedrock)

For side-by-side comparison in the browser with real queries:

```bash
# 1. Configure both backends in .env
cp .env.example .env
# Set ANTHROPIC_API_KEY, and for Bedrock:
# BEDROCK_KNOWLEDGE_BASE_ID=<kb-id>
# BEDROCK_MODEL_ARN=arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6-20251001-v2:0
# BEDROCK_REGION=us-east-1

# 2. Streamlit chat playground (supports all backends + side-by-side comparison)
uv run streamlit run frontend/librarian_chat.py
# → http://localhost:8501 — select any backend or compare multiple side by side
```

### Experiment runner (LangFuse dashboard)

The experiment runner executes all three variants against a golden dataset and optionally logs everything to [LangFuse](https://langfuse.com) for a visual experiment comparison dashboard with per-query traces, per-node latency waterfalls, and aggregated metrics.

```bash
# Run all variants (prints comparison table, logs to LangFuse if enabled)
make eval-experiment

# Or via the CLI directly:
uv run python -m eval.experiment run                     # all variants
uv run python -m eval.experiment run --variant librarian  # single variant
uv run python -m eval.experiment run --path /data/golden.jsonl  # custom dataset
```

**Example output:**
```
  Variant      hit_rate        MRR     n  hits   avg_ms failures
  -------------------------------------------------------------------------
  librarian       1.000      1.000     5     5     0.1 [none]
  raptor          1.000      0.350     5     5     0.1 [none]
  bedrock         1.000      0.350     5     5     0.1 [none]

  Configuration:
    librarian: multilingual k=10 reranker=cross_encoder bm25=0.3/0.7
    raptor: minilm k=5 reranker=passthrough bm25=0.0/1.0
    bedrock: minilm k=5 reranker=passthrough bm25=0.0/1.0
```

**To enable the LangFuse dashboard:**

```bash
# 1. Add credentials to .env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...

# 2. Upload golden dataset to LangFuse (one-time)
uv run python -m eval.experiment upload
# Or with a custom dataset:
uv run python -m eval.experiment upload --path /data/golden.jsonl

# 3. Run experiments — traces are logged to LangFuse
uv run python -m eval.experiment run
```

Once enabled, the LangFuse dashboard shows:
- **Per-node latency waterfalls** — automatic via LangGraph callback handler (condense → analyze → retrieve → rerank → gate → generate)
- **Per-query scores** — hit_rate, MRR, retrieval latency linked to each dataset item
- **Experiment comparison** — side-by-side runs across variants with aggregate metrics
- **Failure clustering** — 13-type taxonomy (zero_retrieval, ranking_failure, coverage_gap, etc.) with remediation suggestions
- **Configuration snapshots** — embedding model, reranker strategy, retrieval k, BM25/vector weights per run

LangFuse tracing is also wired into the **live API routes** — every `/chat` and `/chat/stream` request creates a full trace with per-node spans when `LANGFUSE_ENABLED=true`.

> **Full operational guide**: See [docs/eval-runbook.md](docs/eval-runbook.md) for grader reference, failure clustering taxonomy, observability layers, golden dataset management, and how to add new variants.

---

## Standalone Agents

### Research Agent

Processes PDFs (books, papers, articles) into structured Obsidian notes with YAML frontmatter, wikilinks, tags, and relevance scoring. Handles multi-chapter documents via TOC-aware chunking.

```bash
uv run python -m agents.research path/to/paper.pdf
uv run python -m agents.research path/to/paper.pdf --dry-run   # preview without writing
uv run python -m agents.research path/to/paper.pdf --topic rag  # override topic folder
```

### Presenter (Visualizer)

Interactive presentation authoring agent. Takes a goal/audience/tone, generates an outline (with human approval checkpoint), writes slide content, generates images via Pollinations.ai, and renders a `.pptx` deck.

```bash
uv run python -m agents.visualizer             # full deck workflow
uv run python -m agents.visualizer --image-only # standalone image generation
```

---

## Setup

```bash
# Prerequisites: uv (https://docs.astral.sh/uv/getting-started/installation/)

# Clone and bootstrap
git clone <repo-url> && cd agents
bash setup.sh          # creates .env, installs deps, creates data dirs

# Or manually:
cp .env.example .env   # fill in ANTHROPIC_API_KEY
uv sync --extra librarian --extra api --extra mcp

# Verify the install
make eval-unit
```

### Docker (full stack)

```bash
# API + Streamlit frontend
docker compose -f infra/docker/docker-compose.yml up --build

# With Jaeger tracing
docker compose -f infra/docker/docker-compose.yml --profile tracing up --build
```

Services:
- **API** — `http://localhost:8000` (FastAPI, `/health`, `/chat`, `/chat/stream`, `/ingest`)
- **Frontend** — `http://localhost:8501` (Streamlit)
- **Jaeger UI** — `http://localhost:16686` (tracing profile only)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Claude API key |
| `ANTHROPIC_MODEL_SONNET` | `claude-sonnet-4-6` | Primary model for generation |
| `ANTHROPIC_MODEL_HAIKU` | `claude-haiku-4-5-20251001` | Lightweight model (history condensation, query rewrite) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Embedding model for vector search |
| `RETRIEVAL_STRATEGY` | `chroma` | Vector store backend: `chroma`, `opensearch`, `duckdb`, `inmemory` |
| `RERANKER_STRATEGY` | `cross_encoder` | Reranker: `cross_encoder`, `llm_listwise`, `passthrough` |
| `RETRIEVAL_K` | `10` | Number of chunks to retrieve |
| `RERANKER_TOP_K` | `3` | Chunks passed to generation after reranking |
| `CONFIDENCE_THRESHOLD` | `0.4` | CRAG retry threshold (0–1) |
| `PLANNING_MODE` | `rule_based` | Query analysis mode |
| `CHROMA_PERSIST_DIR` | `.chroma` | ChromaDB storage path |
| `DUCKDB_PATH` | `.duckdb/librarian.db` | DuckDB storage path |
| `BEDROCK_KNOWLEDGE_BASE_ID` | — | AWS Bedrock KB ID (for Bedrock backend) |
| `BEDROCK_MODEL_ARN` | — | Bedrock model ARN |
| `BEDROCK_REGION` | `us-east-1` | AWS region for Bedrock |
| `OTEL_ENABLED` | `false` | Enable OpenTelemetry tracing |
| `LANGFUSE_ENABLED` | `false` | Enable LangFuse observability |
| `READINGS_DIR` | `~/Dropbox/ai_readings` | Source PDF directory (research agent) |
| `OBSIDIAN_VAULT` | `~/workspace/obsidian` | Obsidian vault output (research agent) |

---

## Project Structure

```
src/
  core/                  Shared infra: config, Claude client, structured logging
  librarian/             RAG pipeline core
    config.py              LibrarySettings (pydantic-settings)
    factory.py             DI assembly: create_librarian(), create_ingestion_pipeline()
    bedrock/               AWS Bedrock KB client wrapper
    generation/            Context building, answer generation, prompts
    retrieval/             Embedder/retriever protocols, RRF fusion, caching, scoring
    reranker/              CrossEncoder, LLM listwise, passthrough
    plan/                  Query analysis, intent, expansion, decomposition, routing
    ingestion/             Ingestion pipeline: chunking → embedding → indexing
      chunking/              6 strategies: html_aware, parent_doc, fixed, overlapping,
                             structured, adjacency
      embeddings/            MiniLM + Multilingual e5-large embedders
      parsing/               Cleaning, dedup, enrichment, language detection
    schemas/               Pydantic models: chunks, queries, state, traces
    tasks/                 Golden datasets, synthetic generation, failure clustering
  orchestration/           LangGraph state machine (graph.py) + node subgraphs
  storage/                 Vector DB backends (Chroma, DuckDB, InMemory, OpenSearch),
                           MetadataDB, SnippetDB, TraceDB
  eval/                    Evaluation framework
    variants.py              Three pipeline configs: librarian, raptor, bedrock
    runner.py                EvalRunner orchestrator
    metrics/                 hit_rate@k, MRR, failure clustering
    graders/                 exact_match, llm_judge, RAGAS, DeepEval, MCQ, human
    pipelines/               Regression + capability eval pipelines
  interfaces/              External-facing APIs
    api/                     FastAPI app, routes, middleware, Lambda handler
    mcp/                     MCP servers: librarian, S3, Snowflake
agents/                  Standalone agents (researcher, presenter, cartographer)
frontend/
  librarian_chat.py      Streamlit playground (backend selector, streaming, metadata)
  web/                   Next.js comparison app (side-by-side Librarian vs Bedrock)
infra/
  docker/                Dockerfiles + docker-compose (API, frontend, Jaeger)
  terraform/             AWS deployment: VPC, ALB, ECS, ECR, S3, Lambda
tests/                   Mirrors src/ structure
  librarian/evalsuite/     Regression, capability, and local data eval tests
```

### Console Scripts

| Command | Description |
|---------|-------------|
| `librarian-api` | Start the FastAPI server (uvicorn) |
| `research-agent` | Run the PDF → Obsidian research agent |
| `presenter` | Run the presentation authoring agent |
| `cartographer` | Run session analysis / workflow insights |
| `mcp-librarian` | MCP server for librarian queries + ingestion |
| `mcp-s3` | MCP server for S3 document operations |
| `mcp-snowflake` | MCP server for Snowflake queries |

---

## Stack

- **Python 3.12+**, uv, ruff, pydantic v2, structlog
- **LangGraph** — RAG pipeline orchestration (state machine)
- **Claude API** (anthropic SDK) — all LLM calls (generation, rewriting, grading)
- **Sentence Transformers** — local embedding models (multilingual-e5-large, MiniLM)
- **ChromaDB / OpenSearch / DuckDB** — vector store backends
- **FastAPI** — REST API with SSE streaming
- **Streamlit** — chat playground
- **Next.js 14** — comparison web app (React, Tailwind)
- **AWS Bedrock** — Knowledge Bases integration (boto3)
- **OpenTelemetry + Jaeger** — distributed tracing
- **LangFuse** — LLM observability (optional)
- **Docker + Terraform** — deployment (ECS/Fargate on AWS)
