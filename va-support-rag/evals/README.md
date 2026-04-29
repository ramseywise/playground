# Evaluations (`evals/`)

Offline evaluation for this repo: golden datasets, retrieval checks, answer-quality graders, and optional observability (LangFuse / LangSmith).

## Mental model

| Concept | What it is |
|--------|------------|
| **`EvalTask`** | One evaluation row: `query`, optional `expected_answer`, `metadata` (e.g. expected doc URL), tags. |
| **Graders** | Per-task judges: take an `EvalTask`, return a `GraderResult` (score, pass/fail). Exact match, LLM judge, Ragas, etc. |
| **Metrics** (`evals/metrics/`) | Batch or stage math: hit rate, MRR, reranker stats, confidence-gate calibration. Used by harnesses and `experiment.py`, not interchangeable with “grader” except that regression wraps retrieval numbers into reports. |
| **`evals/tracing.py`** | In-memory `PipelineTrace` / `FailureClusterer` for clustering failures during a run — not LangSmith trace import. |
| **`evals/utils/loaders.py`** | Load golden JSONL / FAQ CSV from disk → `GoldenSample` / `EvalTask`. |
| **`evals/utils/`** | Models (`EvalTask`, reports), settings, protocols. |

**Two ways to run things**

1. **`python -m evals.experiment`** — upload / run retrieval **variants**, score hits (MRR, etc.), optional LangFuse; export JSON with `--export`.
2. **`evals.runner.EvalRunner`** — **capability** (tasks × graders), **regression** (golden tasks × `retrieve_fn` → hit/MRR vs thresholds). Use in tests and custom scripts.

**Observability**

- **LangFuse**: primary integration in `experiment.py` when enabled (see `evals/utils/settings.py`).
- **LangSmith**: `evals/metrics/langsmith_metrics.py` fetches recent root runs (API read). Needs `LANGCHAIN_API_KEY` and the `rag` extra. `make run-dashboard` prints a short table.

## Makefile commands

| Goal | Command |
|------|---------|
| Variant experiment + export JSON | `make run-experiment` |
| Upload golden set to LangFuse (when configured) | `make run-experiment-upload` |
| Recent LangSmith runs (terminal) | `make run-dashboard` |

Dependencies: `uv sync --extra rag`. Env vars: `evals/utils/settings.py`.

## Folder map

| Path | Role |
|------|------|
| `experiment.py` | CLI: upload / run / export (LangFuse-aware when enabled). |
| `runner.py` | `EvalRunner` — capability, regression, static helpers for reranker/gate metrics. |
| `harnesses/` | `run_capability_eval`, `run_regression_eval`. |
| `graders/` | QA grader implementations (`baseline.py` is a good starting point). |
| `metrics/` | Retrieval / reranker / confidence / LangSmith helpers. |
| `utils/` | `models`, `loaders`, `settings`, `protocols`. |
| `tracing.py` | Failure clustering helpers for harness runs. |

## CLI

```bash
uv run python -m evals.experiment --help
```

For code navigation, search for `run_capability_eval`, `run_regression_eval`, or the grader class you need rather than reading `experiment.py` linearly.
