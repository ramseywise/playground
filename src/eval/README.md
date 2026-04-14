# Eval Suite

Evaluation framework for the RAG pipeline. Grades agent responses across
quality dimensions (grounding, completeness, tone, escalation), measures
retrieval/reranker quality, and tracks confidence gate calibration.

## Directory Layout

```
src/eval/
  models.py             Core types: EvalTask, GraderResult, EvalReport
  protocols.py          Grader and GoldenDataset protocols
  runner.py             EvalRunner orchestrator
  loaders.py            JSONL -> GoldenSample loader
  variants.py           Named retrieval config presets
  experiment.py         LangFuse experiment runner (CLI)
  dashboard_data.py     Dashboard data fetcher (LangFuse + local JSON)

  graders/              All grader implementations
    metrics_registry.py   Metric definitions (prompts, thresholds, fields)
    llm_judge.py          Base LLM-as-judge class
    composite_judge.py    Multi-metric single-call judge
    grounding_judge.py    Claim grounding + knowledge override
    completeness_judge.py Multi-part question coverage
    epa_judge.py          Empathy, Professionalism, Actionability
    escalation_judge.py   Escalation appropriateness
    conciseness_grader.py Token budget + optional padding check
    exact_match.py        Exact match + set overlap (deterministic)
    mcq.py                Multiple choice (deterministic)
    human.py              File-based human review queue
    deepeval_grader.py    DeepEval library adapter
    ragas_grader.py       RAGAS library adapter

  harnesses/            Evaluation harnesses
    capability.py         Tasks x graders -> EvalReport
    regression.py         Retrieval hit_rate/MRR regression

  metrics/              Per-stage RAG metrics
    _shared.py            Shared retrieval hit/MRR core loop
    retrieval.py          hit_rate, MRR, precision, recall, NDCG
    reranker.py           Rank displacement, NDCG improvement
    confidence.py         Gate accuracy, FPR/FNR, calibration
```

## Quick Start

```python
from eval.graders import CompositeJudge, GroundingJudge
from eval.runner import EvalRunner
from eval.models import EvalTask, EvalRunConfig

# Multi-metric evaluation (one LLM call)
judge = CompositeJudge(llm, metrics=["grounding", "completeness", "epa"])
result = await judge.grade(task)

# Standalone evaluation
judge = GroundingJudge(llm)
result = await judge.grade(task)

# Full eval run with EvalRunner
runner = EvalRunner(graders=[judge], config=EvalRunConfig(run_name="v1"))
report = await runner.run_capability(tasks)
```

## Graders

### Deterministic (no LLM)

| Class | `grader_type` | What it checks | Pass criteria |
|---|---|---|---|
| `ExactMatchGrader` | `exact_match` | Normalized string equality | exact match |
| `SetOverlapGrader` | `set_overlap` | Token-level Jaccard/F1 | F1 >= threshold |
| `MCQGrader` | `mcq` | Multiple-choice letter match | letter match |

### Hybrid (deterministic + optional LLM)

| Class | `grader_type` | What it checks | Pass criteria |
|---|---|---|---|
| `ConcisenessGrader` | `conciseness` | Token budget ratio + padding | within budget |

### LLM Judges (standalone)

| Class | `grader_type` | What it evaluates | Pass threshold |
|---|---|---|---|
| `GroundingJudge` | `grounding` | Claim-level RAG faithfulness | grounding >= 0.8, no hallucination, parametric <= 0.2 |
| `CompletenessJudge` | `completeness_judge` | Multi-part question coverage | completeness >= 0.7 |
| `EPAJudge` | `epa_judge` | Empathy, Professionalism, Actionability | composite >= 0.65 |
| `EscalationJudge` | `escalation_judge` | Escalation appropriateness | appropriateness == 1.0 |

### CompositeJudge (multi-metric, one LLM call)

Evaluates multiple metrics in a single LLM call. Saves API calls when
you need several quality dimensions per task.

```python
judge = CompositeJudge(llm, metrics=["grounding", "completeness", "epa"])
result = await judge.grade(task)

result.grader_type    # "composite:completeness+epa+grounding"
result.is_correct     # True only if ALL selected metrics pass
result.score          # Mean of per-metric scores
result.dimensions     # {"grounding.grounding_ratio": 0.9, "epa.empathy": 0.8, ...}
```

Available metrics: `grounding`, `completeness`, `epa`, `escalation`.

Use standalone judges when you need granular per-metric failure traces or
independent retry logic.

### Third-party Adapters

| Class | Library | Metrics |
|---|---|---|
| `DeepEvalGrader` | `deepeval` | faithfulness, answer relevancy, contextual precision/recall |
| `RagasGrader` | `ragas` | faithfulness, answer relevancy, context precision/recall |

Both require optional dependencies (`pip install deepeval` / `pip install ragas`).

### Human Review

`HumanGrader` — file-based review queue. Call `submit()` to enqueue
tasks to `pending.jsonl`, then `grade()` reads verdicts from
`completed.jsonl`. Supports structured review tags: `hallucination`,
`retrieval_relevancy`, `tone`, `escalation_failure`, `context_missing`.

## Harnesses

### Capability

```python
from eval.harnesses.capability import run_capability_eval
report = await run_capability_eval(tasks, graders, config=config)
```

Runs every task through every grader. A task passes if ANY grader marks
it correct. Produces breakdowns by category, difficulty, and grader.

### Regression

```python
from eval.harnesses.regression import run_regression_eval, RegressionThresholds
report = await run_regression_eval(
    tasks, retrieve_fn,
    k=5,
    thresholds=RegressionThresholds(hit_rate_floor=0.6, mrr_floor=0.4),
)
```

Evaluates retrieval quality (hit_rate@k, MRR) against golden tasks.
Uses the shared core loop from `metrics/_shared.py`.

## Metrics

Import submodules directly (heavy dependencies are not loaded at package level):

```python
from eval.metrics.retrieval import evaluate_retrieval, precision_at_k, ndcg_at_k
from eval.metrics.reranker import evaluate_reranker, rank_displacement
from eval.metrics.confidence import evaluate_gate, optimal_threshold, calibration_curve
```

### Retrieval

`evaluate_retrieval(golden, retrieve_fn, k)` — hit_rate@k, MRR, failure clustering.
Extended: `precision_at_k`, `recall_at_k`, `ndcg_at_k`.

### Reranker

`evaluate_reranker(queries, k)` — compares pre-rerank (`GradedChunk`) vs
post-rerank (`RankedChunk`) orderings against ground truth.
Per-query: `rank_displacement`, `ndcg_improvement`, `score_correlation`, `top_k_precision_lift`.

### Confidence Gate

`evaluate_gate(scores, truths, threshold)` — gate accuracy, FPR, FNR.
`optimal_threshold(scores, truths)` — finds threshold maximizing F1.
`calibration_curve(scores, truths, n_bins)` — reliability diagram data.

## Data Models

- **`EvalTask`** — input: `id`, `query`, `expected_answer`, `context`, `metadata` (response stored here), `category`, `difficulty`, `tags`
- **`GraderResult`** — output: `task_id`, `grader_type`, `is_correct`, `score`, `reasoning`, `dimensions` (numeric sub-scores), `details` (all extra fields)
- **`EvalReport`** — aggregate: `pass_rate`, `avg_score`, `by_category`, `by_difficulty`, `by_grader`, `failure_clusters`
- **`EvalRunConfig`** — reproducibility snapshot: `model_id`, `prompt_version`, `corpus_version`, `top_k`

## Adding a New LLM Metric

1. Add a `MetricDefinition` entry to `graders/metrics_registry.py` with standalone/composite prompts, required fields, and pass predicate
2. Create a thin `LLMJudge` subclass in its own file importing from the registry
3. Export from `graders/__init__.py`
4. Add tests in `tests/eval/graders/`

The new metric is automatically available in `CompositeJudge` via its registry name.

## Running

```bash
# Full eval test suite
uv run pytest tests/eval/ -v

# Experiment CLI
uv run python -m eval.experiment upload           # seed LangFuse dataset
uv run python -m eval.experiment run              # run all variants
uv run python -m eval.experiment run --variant librarian --export results.json
```
