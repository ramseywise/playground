# Research: Three-Way RAG Architecture Comparison

Date: 2026-04-12 (updated)
Context: Decision support for production architecture across three options

| | BookKeeper Hero (Option 0) | PoC 1 — Bedrock KB | PoC 2 — Librarian (LangGraph) |
|---|---|---|---|
| Retrieval | Vendor black box | Dense-only (Titan/Cohere + OpenSearch Serverless) | Hybrid BM25+vector + cross-encoder rerank + CRAG |
| Generator | Unknown (probably vendor-hosted) | Gemini (via API) | Claude (Anthropic API) |
| Reranker | Unknown | None (sticky note: "no reranker — 15% uplift in precision" gap) | `ms-marco-MiniLM-L-6-v2` cross-encoder |
| Observability | Thumbs up/down only | CloudWatch + response text | Structured traces, confidence scores, LangFuse/LangSmith |
| Corpus control | None | S3 + AWS console | Full: 6 chunking strategies, swappable embedder, metadata schema |
| Integration complexity | Low (SaaS embed) | Medium (AWS managed, but 2 vendors: Bedrock + Gemini) | High (self-managed pipeline + infra) |
| Vendor lock-in | High (third-party SaaS) | High (AWS data plane + Gemini) | Low (Chroma portable, open-source orchestration) |
| Self-learning loop | No (vendor-side only) | No | Yes (failure attribution → eval harness → CI regression) |

---

## Risk scorecard

🟢 Low risk &nbsp; 🟡 Medium risk &nbsp; 🔴 High risk

| Risk dimension | BookKeeper Hero | PoC 1 — Bedrock KB | PoC 2 — Librarian |
|---|---|---|---|
| **Integration complexity** | 🟢 SaaS embed, no infra | 🟡 AWS managed but two vendors (Bedrock + Gemini) — two auth flows, two failure domains, no unified trace | 🔴 Full pipeline ownership: embedder, vector store, reranker, LangGraph graph, LangFuse — weeks to stand up |
| **Cost** | 🟡 SaaS license (fixed, opaque, likely high at scale) | 🟢 Serverless per-call (~$0.0005–0.001/query + model tokens); cheapest at low volume (<5K/month) | 🟡 Fixed infra ~$50–80/month (Fargate) + model tokens; cheaper than Bedrock above ~5K queries/month |
| **Latency** | 🟡 Unknown; vendor-controlled; no streaming | 🔴 1–2.5s blocking (single `RetrieveAndGenerate` call, no streaming); feels slow even when fast | 🟢 ~800ms–1.5s TTFT with streaming; reranker adds 200–500ms but user sees tokens arriving |
| **Hallucination / answer quality** | 🔴 Unknown risk, no visibility; no reranker, no confidence gate, no failure attribution | 🔴 High risk: dense-only retrieval misses jargon/code; no reranker (15% precision gap); no confidence gate; no CRAG retry | 🟢 Lowest risk: hybrid retrieval + cross-encoder reranker prunes noise; CRAG retries on low confidence; `HistoryCondenser` prevents multi-turn drift |

---

## Extended risk surface

### Engineering risks

| Risk | BookKeeper Hero | PoC 1 — Bedrock KB | PoC 2 — Librarian |
|---|---|---|---|
| **Cold start** | 🟢 Vendor-managed, always warm | 🟢 Serverless, no model to load | 🔴 Local embedder (560MB) needs loading on Fargate cold start — 30–60s spike on first request after scale-to-zero |
| **Ingest concurrency** | 🟢 Vendor-managed | 🟢 S3 sync is parallelisable | 🟡 Chroma has a single-process write lock — parallel ingest workers will contend; mitigation: batch ingest via single worker or switch to OpenSearch |
| **Rate limiting / throttling** | 🟡 One vendor SLA | 🔴 Two independent throttle surfaces — Bedrock service quotas + Gemini rate limits; a Gemini spike doesn't show up in Bedrock metrics | 🟡 One external API (Anthropic); local components don't throttle |
| **Model drift** | 🔴 Vendor updates silently; no change log; you find out from user complaints | 🔴 Gemini model versions update on Google's schedule; Bedrock model ARN is pinned but KB embedding model may not be | 🟢 Model versions explicitly pinned in config; you control the upgrade |
| **Data residency** | 🔴 Corpus + queries sent to third-party vendor; GDPR/SOC2 depends on their controls | 🔴 Queries sent to Gemini (Google); corpus lives in AWS OpenSearch Serverless | 🟡 Only generation context sent to Anthropic; embedder is local; Chroma is local — smallest external footprint |
| **Re-embedding cost on model swap** | 🔴 Full re-ingest through vendor pipeline; opaque cost and timeline | 🟡 Full re-embed required; AWS hides the cost until you trigger it | 🟡 Full re-embed required but explicit: `IngestionPipeline` with new `EMBEDDING_PROVIDER`; cost is known upfront |
| **Multi-turn checkpoint storage** | 🟢 Vendor-managed | 🟡 `sessionId` param; AWS manages history storage | 🔴 LangGraph checkpointing needs a persistent store (Redis or Postgres) at prod scale — not wired yet; in-memory checkpointer works locally only |
| **Bus factor / knowledge risk** | 🟢 Vendor maintains it | 🟡 AWS-managed, but internal Gemini wiring and corpus shape is internal knowledge | 🔴 LangGraph graph topology, chunker configs, eval harness, and LangFuse setup all require deep familiarity — high knowledge concentration risk if team changes |

---

### Product and design risks

| Risk | BookKeeper Hero | PoC 1 — Bedrock KB | PoC 2 — Librarian |
|---|---|---|---|
| **Citation fidelity** | 🟡 Unknown format; vendor-controlled | 🟡 Document-level source links — user may click and land on the wrong section of a long doc | 🟢 Chunk-level citations with `confidence_score`; can surface the exact passage and flag low-confidence answers |
| **Escalation path** | 🟢 Built-in (explicit escalation button in diagram) | 🔴 No escalation logic; low-confidence answers go to the user as if they were high-confidence | 🟡 `confidence_score` output exists but not yet wired to human handoff — product gap to design and implement |
| **Out-of-scope UX** | 🟡 Vendor-defined deflection message | 🔴 Dense retrieval will surface the closest match and the model will confabulate rather than deflect | 🟢 `QueryRouter` routes out-of-scope to a controlled "I don't know" path — needs UX copy but the gate exists |
| **Personalization ceiling** | 🔴 No account/CRM context; vendor can't access your data model | 🔴 Static corpus only; no tool use, no CRM integration | 🟢 LangGraph tool nodes can call internal APIs (billing, CRM, account data) — "what's my invoice?" is a graph extension, not a rearchitecture |
| **Answer consistency** | 🔴 Same question, different session → potentially different answer; no control or visibility | 🔴 Same problem; Bedrock generation is non-deterministic and opaque | 🟡 Same underlying variance but `retrieved_chunks` and `confidence_score` are logged — inconsistency is diagnosable and reproducible in eval |
| **Prompt injection / jailbreak** | 🟡 Vendor guardrails exist; strength unknown and not auditable | 🟡 Claude's built-in safety applies to generation; retrieval layer is not hardened | 🔴 Your responsibility end-to-end — system prompts in `generation/prompts.py` are the only guardrail; need explicit adversarial testing |
| **Feedback loop closure** | 🔴 Thumbs up/down captured by vendor; no path to corpus or retrieval improvement | 🔴 CloudWatch logs exist but no structured signal; no path from user feedback to retrieval fix | 🟢 Full loop: UI signal → trace classification → failure attribution → grounding eval → CI regression — but requires intentional wiring at each step |

---

## Key question: off-the-shelf observability vs self-learning pipeline

The diagram you're working from draws a feedback loop that goes:

```
UI (thumbs up/down, escalation, chat history)
  → V2 Eval Suite: Classify Traces
      → Grounding Pipeline (regression tests)
      → Failure-Attribution Pipeline (failure taxonomy → capability tests → confidence graders)
          → pytest / CI integration
              → Eval Harness (trials × graders × metrics)
```

This loop is only achievable end-to-end with **PoC 2**. Here's why each option breaks down:

**BookKeeper Hero:** Feedback signal (thumbs up/down, escalation) exists but is trapped in a vendor dashboard. You can't route those signals back into a retrieval eval harness or attribute failures to retrieval vs. generation — the pipeline internals are invisible.

**PoC 1 (Bedrock KB):** You get response text + citations. There is no `confidence_score`, no `retrieved_chunks` list, no per-step timing. "Why did it hallucinate?" has no answer. You can build an eval harness that scores final answers, but you can't diagnose whether the failure was a retrieval miss, wrong chunk, or model drift. The Grounding and Failure-Attribution pipelines in the diagram become guesswork.

**PoC 2 (Librarian):** Every node emits structured state (`retrieved_chunks`, `reranked_chunks`, `confidence_score`, `failure_reason`). The `FailureClusterer` can group misses by pattern (`zero_retrieval`, `expected_doc_not_in_top_k`, `low_confidence`). This feeds directly into:
- Grounding pipeline: regression tests assert that specific docs appear in top-k for known queries
- Failure-attribution: classify failure by stage (retrieval miss vs. reranker fail vs. model hallucination)
- Eval harness: trials are repeatable because the full intermediate state is logged
- CI: a new corpus ingest or chunking config change triggers the eval suite and fails fast

The self-learning loop — where observed failures improve the system without manual intervention — is what separates PoC 2 from the others architecturally. Off-the-shelf gets you signals; the custom pipeline lets you act on them.

---

## Summary

Bedrock KB is the right default for launch: zero infra, fast to wire, and acceptable quality
for well-structured corpora. The Librarian pipeline wins on every quality dimension that
matters once you hit production — retrievability gaps, multi-turn accuracy, and failure
diagnosis. The side-by-side switch is already implemented in the UI; test both before committing.

---

## Head-to-head

### 1. Retrieval quality

**Bedrock KB**
- AWS manages embedding (Titan or Cohere), chunking, and vector store (OpenSearch Serverless)
- Single-strategy: dense vector retrieval only — no BM25 hybrid, no keyword fallback
- Fixed chunking: AWS auto-chunks your S3 documents; no heading-aware or parent-doc strategies
- No reranking step — top-k by cosine similarity is the final ranking

**Librarian**
- Hybrid BM25 + vector with configurable weights (default 0.3/0.7)
- RRF (Reciprocal Rank Fusion) across multi-query variants — expands the query into N reformulations and merges ranked lists
- Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) scores each (query, chunk) pair and reorders — consistently 10-20% hit_rate improvement vs cosine-only ranking
- CRAG retry loop: if confidence score falls below threshold, rewrites the query and retrieves again before generating

**Verdict:** Bedrock KB retrieval is a black box optimised for average-case corpora. Librarian's
hybrid + rerank + CRAG combination is measurably better for domain-specific or technical corpora
where term overlap matters (code, product names, version numbers, jargon).

---

### 2. Multi-turn accuracy

**Bedrock KB**
- Has `sessionId` parameter that passes conversation history to the model
- AWS handles context injection — limited control over how history is used
- No explicit query reformulation: "what about the Python version?" sent verbatim to the retriever

**Librarian**
- `HistoryCondenser` node rewrites the latest user query to be self-contained given prior turns
- "What about the Python version?" → "What is the Python version for [topic from prior turn]?"
- Only fires on multi-turn (single-turn has zero added latency)
- Uses Haiku (~$0.001/rewrite) — cheap, fast, targeted

**Verdict:** Bedrock KB's session handling will silently degrade on coreference-heavy
conversations ("that one", "and for the other method?"). The Librarian's condenser prevents
retrieval misses that look like hallucinations but are actually bad queries.

---

### 3. Observability and failure diagnosis

**Bedrock KB**
- Response: answer text + citations list. That's it.
- "Why did it give a wrong answer?" → no answer. Was it a retrieval miss? Wrong chunk? Model drift?
- No intermediate state, no scores, no ranked list, no per-step timing

**Librarian**
- Every node emits structured log events: `graph.crag.retry`, `chroma.search.done`, `reranker.done`
- `confidence_score` (0–1) tells you how much the reranker trusted its own output
- `retrieved_chunks`, `graded_chunks`, `reranked_chunks` all in state — inspectable at any step
- `failure_reason` in eval traces: `zero_retrieval`, `expected_doc_not_in_top_k`, `low_confidence`
- `FailureClusterer` groups misses by pattern for batch diagnosis
- Optional LangFuse integration for trace-level debugging and metric dashboards

**Verdict:** When Bedrock KB fails, you can't tell why. When Librarian fails, you can diagnose
it to the exact node and fix it. This compounds over time — the Librarian pipeline improves;
Bedrock KB failures stay mysteries.

---

### 4. Latency

| Step | Bedrock KB | Librarian (warm, streaming) |
|---|---|---|
| Embed query | ~100ms (AWS-managed) | ~100–200ms (local) or ~50ms (Voyage API) |
| Vector retrieve | ~200ms (OpenSearch Serverless) | ~50ms (Chroma local) or ~100ms (OpenSearch) |
| Rerank | None | ~200–500ms (cross-encoder CPU) |
| Generate (Claude) | Included in API call | ~400–800ms TTFT |
| **Total** | **1–2.5s (blocking, no stream)** | **~800ms–1.5s TTFT (streaming)** |

Bedrock KB's `RetrieveAndGenerate` is a single blocking call — no streaming token delivery.
Librarian streams from the generation node, so the user sees text arriving within ~1s.

**Verdict:** Bedrock KB latency is acceptable but feels slower because it's a blocking wall.
Librarian's streaming gives a significantly better perceived latency even if wall-clock is similar.

---

### 5. Cost structure

**Bedrock KB**
- Bedrock KB retrieval: ~$0.0004–0.001 per query (OpenSearch Serverless unit costs)
- Titan/Cohere embedding: ~$0.0001 per query
- Claude model tokens: same as Librarian
- No fixed infra cost — fully serverless
- Gets expensive at sustained high volume (>10K queries/day)

**Librarian on ECS/Fargate**
- Fixed: ~$50–80/month for always-on 2vCPU/4GB task
- Variable: Claude model tokens only (no embedding API cost with local model)
- Cheaper at volume (fixed cost amortises), more expensive at low volume
- Swapping local embedder for Voyage AI (~$0.00012/1K tokens) adds ~$0.0001/query but
  eliminates the 560MB RAM tax and model cold-start

**Verdict:** Bedrock KB wins at low volume. Librarian wins above ~5K queries/month (fixed
Fargate cost becomes cheaper per query than Bedrock's per-call fees).

---

### 6. Corpus control

**Bedrock KB**
- Ingest: upload documents to S3, configure KB in AWS console, Bedrock handles the rest
- Chunking: fixed-size or hierarchical (limited options, no heading-aware strategy)
- Embedding: Titan v2 (1536-dim) or Cohere embed — not swappable without re-indexing
- Metadata filtering: supported, but schema is AWS-defined
- Re-ingestion: full re-sync via S3 sync job

**Librarian**
- Ingest via `IngestionPipeline`: chunk → embed → index → MetadataDB + SnippetDB
- Six chunking strategies: `html_aware` (heading-boundary), `parent_doc`, `fixed`, `overlapping`, `structured`, `adjacency` — swappable via config
- Embedding model: swappable via `EMBEDDING_PROVIDER` env var; currently local e5-large
- Metadata schema: fully custom via `ChunkMetadata` (namespace, topic, access_tier, etc.)
- Selective re-ingestion by `doc_id` without full re-index

**Verdict:** Bedrock KB's corpus control is acceptable for document sets you don't iterate on.
For corpora that change frequently or require fine-grained control (access tiers, namespaces,
language-specific chunking), Librarian's ingestion pipeline is essential.

---

### 7. Vendor lock-in

**Bedrock KB**
- Data plane: AWS OpenSearch Serverless (proprietary)
- Vector store schema: AWS-managed, not portable
- Embedding format: Titan/Cohere vectors in AWS — can't move to another provider without re-embedding
- Switching cost: high — need to re-ingest, re-embed, rebuild KB

**Librarian**
- Vector store: Chroma (local, portable) or OpenSearch (self-managed) — swap via `RETRIEVAL_STRATEGY` env var
- Embedding: local model or any cloud API (Voyage, Cohere, OpenAI) via `EMBEDDING_PROVIDER`
- LangGraph: open source, runs anywhere Python runs
- Switching cost: low — change config, re-ingest to new backend

**Verdict:** Bedrock KB commits you to AWS's data plane. Librarian is portable — can run on
any cloud or on-prem.

---

### 8. Integration complexity

| Dimension | BookKeeper Hero | PoC 1 — Bedrock KB | PoC 2 — Librarian |
|---|---|---|---|
| Setup effort | Hours (SaaS embed) | Days (S3 + KB config + Gemini wiring) | Weeks (pipeline + infra + eval harness) |
| Dependencies | 1 vendor contract | AWS + Gemini API | Python stack, Chroma/OpenSearch, LangFuse/LangSmith |
| Corpus ingest | Vendor UI | S3 sync + AWS console | `IngestionPipeline` CLI + config |
| Config surface | None | KB ID + model ARN + S3 bucket | env vars for each stage (embedder, chunker, retriever, reranker) |
| Failure surface | Vendor-managed | AWS service health + Gemini uptime | Each pipeline node is a failure point; mitigated by node-level error handling |
| On-call burden | Low (vendor SLA) | Medium (AWS + Gemini independently) | Higher (own the stack), offset by observability |
| Iteration speed | None (closed) | Slow (re-ingest to change chunking) | Fast (swap config, re-ingest selectively by `doc_id`) |

**Key integration risk in PoC 1:** Two-vendor answer chain (Bedrock retrieval + Gemini generation) means two failure domains, two pricing models, two auth flows, and no unified trace. If Gemini drifts, you don't know if the problem is retrieval or generation.

---

### 9. Hallucination risk

Hallucination in RAG has two sources: **retrieval misses** (model lacks grounding) and **context overload** (model ignores good chunks or confabulates from noise). Both need to be measured independently.

| Risk factor | BookKeeper Hero | PoC 1 — Bedrock KB | PoC 2 — Librarian |
|---|---|---|---|
| Retrieval miss rate | Unknown | High (dense-only, no keyword fallback for jargon/code) | Lower (hybrid + CRAG retry) |
| Context noise | Unknown | High (no reranker — top-k cosine only) | Lower (cross-encoder reranker prunes irrelevant chunks) |
| Low-confidence generation | No gate | No gate | CRAG gate: confidence < threshold → retry before generating |
| Multi-turn drift | Unknown | Likely (raw query sent to retriever) | Low (`HistoryCondenser` rewrites ambiguous queries) |
| Diagnosable? | No | No | Yes (`failure_reason`, `confidence_score`, chunk-level traces) |

The PoC 1 sticky note in the diagram ("No reranker — 15% uplift in precision") quantifies the direct quality gap: every answer from PoC 1 is generated from a noisier context than PoC 2. That noise manifests as hallucination when the model synthesises across irrelevant chunks or fills gaps that the retriever left.

---

## When each option is the right choice

**BookKeeper Hero (Option 0 — off-the-shelf)**
- Time-to-live is the only constraint — need something in front of users in days
- Org has no ML/platform engineering capacity to maintain a pipeline
- Feedback loop requirements are minimal (thumbs up/down sufficient)
- Acceptable to not own the data or the improvement trajectory

**PoC 1 — Bedrock KB**
- Corpus is stable and well-structured (standard prose, not code or jargon-heavy)
- No budget for always-on infra — serverless, zero fixed cost
- Already committed to AWS (VPC, IAM, existing OpenSearch)
- Low query volume (< 5K/month) — per-call cost cheaper than fixed Fargate
- Acceptable to accept black-box retrieval without failure diagnosis
- Note: Gemini as the generator in the PoC diagram adds vendor complexity — consider swapping to Claude via Bedrock to reduce integration surface

**PoC 2 — Librarian (LangGraph)**
- Retrieval quality matters — technical docs, product names, version numbers, code
- Multi-turn conversations are a core UX (coreference resolution critical)
- You need to improve the system over time — can't improve what you can't observe
- The full feedback loop (UI signals → trace classification → grounding eval → CI regression) is a requirement, not a nice-to-have
- Higher volume (> 5K queries/month) — fixed Fargate cost amortises
- Corpus requires custom chunking, metadata, or access tiers
- Streaming perceived latency matters to UX

---

## Side-by-side testing (already implemented)

The Streamlit frontend at `frontend/librarian_chat.py` has a radio button switching between
`"Python RAG (Librarian)"` and `"AWS Bedrock KB"`. The API routes dispatch on
`req.backend: Literal["librarian", "bedrock"]`.

**To activate Bedrock KB testing:**
```bash
# .env
BEDROCK_KNOWLEDGE_BASE_ID=<kb-id-from-aws-console>
BEDROCK_MODEL_ARN=arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6-20251001-v2:0
BEDROCK_REGION=us-east-1
```

Same corpus ingested to both backends → same queries → compare `confidence_score`,
`citations`, response quality, and latency side by side in the Streamlit UI.

**Key things to test:**
- Multi-turn queries with coreference ("what about the Python one?")
- Technical/jargon queries (version numbers, product names, code terms)
- Queries where the answer spans multiple documents
- Edge cases: out-of-scope queries, ambiguous queries
