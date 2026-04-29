# VA Eval Base — Shared Evaluation Harness

Unified evaluation framework for comparing all three VA implementations:
- **va-google-adk** — Google ADK multi-agent
- **va-langgraph** — LangGraph StateGraph
- **va-support-rag** — RAG-only Q&A service

## Architecture

### Layers

1. **Models** (`models.py`) — Shared data structures:
   - `EvalTask` — Single Clara ticket or test case
   - `ServiceResponse` — Normalized response from any VA service
   - `GraderResult` — Score from a single grader on a single task

2. **Harness** (`harness.py`) — HTTP transport:
   - Sends Clara tickets to all 3 gateways concurrently
   - Streams SSE responses from va-google-adk & va-langgraph
   - Calls va-support-rag REST endpoint synchronously
   - Normalizes all responses into `ServiceResponse`

3. **Graders** (`graders.py`) — Baseline metrics shared across all services:
   - `SchemaGrader` — validates response structure (AssistantResponse)
   - `MessageQualityGrader` — checks message is non-empty and reasonable length
   - `RoutingGrader` — for orchestration services, checks routing accuracy

4. **Metrics** (`metrics.py`) — Service-specific layers:
   - `RAGMetricsGrader` — retrieval quality, citations, escalation
   - `OrchestrationMetricsGrader` — routing, suggestions, navigation

5. **Runner** (`runner.py`) — Orchestration:
   - Loads 278 Clara fixtures
   - Runs all tasks on all services (concurrent)
   - Applies all graders in parallel
   - Aggregates and formats results

## Usage

### Basic Eval (Baseline Graders Only)

```bash
uv run python -m va_eval_base.cli --baseline-only
```

### Full Eval (Baseline + Service-Specific)

```bash
uv run python -m va_eval_base.cli
```

### Save Results to JSON

```bash
uv run python -m va_eval_base.cli --output results/eval-2026-04-29.json
```

### Custom Run Name

```bash
uv run python -m va_eval_base.cli --name "e1-reranker-top-k-3"
```

## Making it a Makefile Target

Add to root `Makefile`:

```makefile
.PHONY: va-eval-baseline
va-eval-baseline:
	cd va-eval-base && uv run python -m cli --output ../results/baseline-$(shell date +%s).json
```

Then:

```bash
make va-eval-baseline
```

## Results Format

Human-readable summary printed to stdout:

```
================================================================================
Eval Report: baseline-eval
Timestamp: 2026-04-29T22:15:00.123456+00:00
================================================================================

Overall: 245/278 passed (88.1%)
Average Score: 0.851

By Service:
  va-google-adk
    Pass Rate: 85/278 (30.6%)
    Avg Score: 0.823
      schema: 85/278 (30.6%) — 1.000
      message_quality: 85/278 (30.6%) — 0.892
      routing: 78/278 (28.1%) — 0.765
      orchestration_metrics: 80/278 (28.8%) — 0.702

  va-langgraph
    Pass Rate: 92/278 (33.1%)
    Avg Score: 0.867
      ...

  va-support-rag
    Pass Rate: 68/278 (24.5%)
    Avg Score: 0.812
      schema: 68/278 (24.5%) — 1.000
      message_quality: 68/278 (24.5%) — 0.845
      rag_metrics: 62/278 (22.3%) — 0.723
      ...
```

Full JSON output saved if `--output` is specified.

## Integration with Other Evals

This harness is **baseline-only** by design. Service-specific evals can fork off:

- **va-langgraph**: Add LLM-as-judge graders (already in `va-langgraph/eval/graders/`)
- **va-google-adk**: Create `va-google-adk/tests/evalsuite/` with ADK-specific graders
- **va-support-rag**: Extend with retrieval metrics (hit_rate, MRR, reranker score)

## Extending Baseline Graders

To add a new baseline grader:

1. Subclass `BaselineGrader`
2. Implement `async def grade(task, response) -> GraderResult`
3. Add to `runner.py`: `graders.append(YourNewGrader())`

Example:

```python
class LengthBalanceGrader(BaselineGrader):
    grader_type = "length_balance"

    async def grade(self, task: EvalTask, response: ServiceResponse) -> GraderResult:
        msg = response.message
        query = task.query
        ratio = len(msg) / max(1, len(query))
        is_correct = 1.0 < ratio < 20.0  # response is reasonably longer than query
        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            service=response.service,
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            reasoning=f"Response/query length ratio: {ratio:.2f}",
            dimensions={"ratio": ratio},
        )
```

## Concurrency & Performance

- Clara fixtures: 278 tasks
- Services: 3 (concurrent)
- Graders per task: 5 baseline + 2 service-specific (7 total)
- **Total grader tasks**: 278 × 3 × 7 = 5,838

With concurrency tuning (harness semaphore + asyncio):
- ~30–45 sec for full eval run (depends on service latency)

## Fixtures

Clara German customer service tickets, stratified by CES (Customer Effort Score):

- **278 tickets** across 6 intent domains
- **1 (easy)** to **7 (high effort)** rating
- Sourced from real Billy customer support
- Expected intents: invoice, quote, customer, product, expense, banking, accounting, insights, support

Loaded from: `va-langgraph/tests/evalsuite/fixtures/clara_tickets.json`
