# Plan: VA Agent Improvements — Tool × Domain Roadmap

> Status: Revised — 2026-04-20
> Scope: `playground/va-google-adk` + `playground/va-langgraph`
> Framing: tool-function × domain perspective; each new tool directly addresses
> one or more BA discovery questions from the feature inventory.

---

## 0. Tool backend architecture — dev vs prod

The tool layer is the **abstraction boundary**. Function signatures never change;
the backend that executes them does.

```
Tool interface (function signatures + docstrings)
              ↓                             ↓
       Dev / CI / staging            Production
   Billy MCP stub server         Real Billy REST API
   (FastMCP → SQLite)            (httpx + OAuth + org_id)
   adk-agent-pocs/               va-google-adk / va-langgraph
     mcp_servers/billy/            shared/tools/ backed by
                                   direct httpx calls
```

**Dev stub** (`mcp_servers/billy/`) — already exists, 110 tests passing:
- MCP server on `:8765` (`main_noauth.py` — stdio or SSE)
- REST API on `:8766` (`main.py` — FastAPI, Swagger at `/docs`)
- SQLite `billy.db` — persistent, seeded, shared across both servers
- Controlled by `BILLY_DB` env var; `reset_db.py` reloads full fixture

**Production path** (future, not blocking dev) — same tool signatures, different
implementation: thin httpx wrapper around the real Billy REST API, OAuth token
from session state, `organisation_id` injected per-request. Controlled by
`BILLY_BACKEND=api` env var when that layer is built.

**Consequence for the build order**: Phase 7 from the previous plan ("replace stubs
with real API") was wrong. The MCP stub is *permanent dev infrastructure*, not
a temporary placeholder. Production is a separate implementation of the same
interface, built when we have real API access.

---

## 1. Current state at a glance

### 1a. va agents (in-memory Python dicts — not wired to MCP)

| Domain | Tools today | Capability type |
|--------|------------|-----------------|
| Invoices | `get_invoice`, `list_invoices`, `get_invoice_summary`, `create_invoice`, `edit_invoice` | Ask + Execute |
| Quotes | `list_quotes`, `create_quote`, `create_invoice_from_quote` | Ask + Execute |
| Customers | `list_customers`, `create_customer`, `edit_customer` | Ask + Execute |
| Products | `list_products`, `create_product`, `edit_product` | Ask + Execute |
| Emails | `send_invoice_by_email`, `send_quote_by_email` | Act |
| Invitations | `invite_user` | Act |
| Support | `fetch_support_knowledge` | Guide |
| **Expenses** | — | MISSING |
| **Banking** | — | MISSING |
| **Insights** | — | MISSING |
| **Accounting** | — | MISSING |

State mutations do not persist between agent restarts (in-memory). No insight
analytics. Quotes domain not in MCP server at all (va-specific addition).

### 1b. MCP stub server — what already exists

Registered in MCP (`common.py`):

| Tool | Domain |
|------|--------|
| `list_customers`, `create_customer`, `edit_customer` | Customers |
| `get_invoice`, `list_invoices`, `get_invoice_summary`, `create_invoice`, `edit_invoice` | Invoices |
| `get_insight_monthly_revenue`, `get_insight_top_customers` | Insights (partial) |
| `list_products`, `create_product`, `edit_product` | Products |
| `send_invoice_by_email` | Emails |
| `invite_user` | Invitations |
| `fetch_support_knowledge` | Support |

Implemented in `invoices.py` but **NOT yet registered in MCP** (REST-only):

| Tool | What it does |
|------|--------------|
| `get_insight_revenue_summary` | KPI cards: invoiced / collected / outstanding / overdue with YoY delta |
| `get_insight_invoice_status` | Status breakdown: draft / unpaid / paid / overdue |
| `get_insight_aging_report` | AR aging buckets: Current / 1–30 / 31–60 / 61–90 / 90+ days |
| `get_insight_customer_summary` | Per-customer KPIs + open invoice list |
| `get_insight_product_revenue` | Products ranked by revenue with quantity sold |
| `get_invoice_lines_summary` | Revenue per product (invoice lines join) |

Missing from MCP server entirely (no domain, no table):
- Quotes (`list_quotes`, `create_quote`, `create_invoice_from_quote`, `send_quote_by_email`)
- Expenses
- Banking
- Accounting

---

## 2. Tool × domain gap matrix

### 2.1 Gaps within existing domains

These all go into the **MCP server first**, then the agents consume them via MCP.

**Invoices** — missing from both MCP and agents:

| Tool | Signature | Addresses |
|------|-----------|-----------|
| `get_invoice_dso_stats` | `(contact_id?, year?) → {avg_days_to_pay, trend, overdue_rate}` | Are payments getting slower? |
| `send_invoice_reminder` | `(invoice_id, message?) → {sent: bool}` | Collections tightening |
| `void_invoice` | `(invoice_id, reason) → {voided: bool}` | Invoice lifecycle — cancel/void |

Note: `get_revenue_by_customer` and `get_revenue_by_product` are already covered by
`get_insight_top_customers` and `get_insight_product_revenue` — just need MCP registration.

**Quotes** — missing entirely from MCP server (no schema, no table):

| Tool | Signature | Addresses |
|------|-----------|-----------|
| `list_quotes` | `(states?, contact_id?, page?) → {quotes, total}` | Pipeline visibility |
| `create_quote` | `(contact_id, lines, expiry_days?) → quote` | Quote authoring flow |
| `edit_quote` | `(quote_id, contact_id?, lines?, expiry_days?) → quote` | Quote lifecycle |
| `create_invoice_from_quote` | `(quote_id) → invoice` | Quote-to-revenue conversion |
| `send_quote_by_email` | `(quote_id, contact_id, subject, body) → {sent: bool}` | Quote delivery |
| `get_quote_conversion_stats` | `(year?) → {sent, accepted, declined, conversion_rate}` | Pipeline health |

Requires adding a `quotes` table to `db.py` and a new `tools/quotes.py` in the MCP server.

**Customers** — missing from MCP:

| Tool | Signature | Addresses |
|------|-----------|-----------|
| `get_customer` | `(customer_id) → full record` | Single customer detail |

Note: `get_customer_revenue_summary` is already `get_insight_customer_summary` — needs
MCP registration.

**Products** — missing from MCP:

| Tool | Signature | Addresses |
|------|-----------|-----------|
| `get_product` | `(product_id) → full record with price` | Single product detail |

Note: `get_product_revenue_summary` is already `get_insight_product_revenue` — needs
MCP registration.

---

### 2.2 Expenses domain — single highest-leverage item

Unlocks the entire **Profitability**, **Break-even**, and **Costs** discovery clusters
(≈12 questions currently gapped). All downstream domains (Banking runway, Insights margin)
are blocked on this.

All tools go into the MCP server (`app/tools/expenses.py` + `quotes` table in `db.py`):

| Tool | Signature | Addresses |
|------|-----------|-----------|
| `list_expenses` | `(date_from?, date_to?, category?, vendor?, page?) → {expenses, total}` | What did I spend on X? |
| `get_expense` | `(expense_id) → full record` | Expense detail |
| `create_expense` | `(vendor, amount, date, category?, description?, vat_amount?) → expense` | Log a new expense |
| `get_expense_summary` | `(year?, period?) → {total, by_category: [{cat, amount, pct}]}` | Total by category; MoM shift |
| `get_vendor_spend` | `(vendor?, year?) → [{vendor, total, count}]` | Vendor audit |
| `get_expenses_by_category` | `(year?) → [{category, total, fixed_or_variable}]` | Fixed vs variable cost split |
| `get_gross_margin` | `(year?, period?) → {revenue, cogs, gross_margin_pct}` | Are COGS creeping up? |

`fixed_or_variable` classification: start with a user-settable flag on the category
(column in the DB); LLM inference at query time is a Phase 6+ enhancement.

---

### 2.3 Banking domain

Blocked on Expenses (burn rate requires expense history). Build after Expenses.

| Tool | Signature | Addresses |
|------|-----------|-----------|
| `get_bank_balance` | `() → [{account, balance, currency}]` | Current balance; cashflow health |
| `list_bank_transactions` | `(date_from?, date_to?, account_id?, page?) → {transactions, total}` | Upcoming payments |
| `match_transaction_to_invoice` | `(transaction_id, invoice_id) → {matched: bool}` | Bank reconciliation |
| `get_cashflow_forecast` | `(months?) → [{month, projected_inflow, projected_outflow, net}]` | 90-day forecast |
| `get_runway_estimate` | `() → {balance, avg_monthly_burn, runway_months}` | How many months of runway? |

---

### 2.4 Insights domain — mostly already built, needs wiring

6 of 8 insight tools exist in `invoices.py` but are not registered in MCP.
Cross-domain tools (margin, break-even) are new — blocked on Expenses being in the DB.

**Already built, need MCP registration only:**
`get_insight_revenue_summary`, `get_insight_invoice_status`, `get_insight_aging_report`,
`get_insight_customer_summary`, `get_insight_product_revenue`, `get_invoice_lines_summary`

**New — depend on Expenses data:**

| Tool | Signature | Addresses |
|------|-----------|-----------|
| `get_net_margin` | `(period?) → {revenue, total_costs, net_profit, net_margin_pct}` | After paying everything, what's my net margin? |
| `get_margin_by_product` | `(year?) → [{product, revenue, cogs, margin_pct}]` | Which offers have the highest margin? |
| `get_customer_concentration` | `(year?) → {top_1_pct, top_3_pct, hhi}` | Am I over-dependent on a few clients? |
| `get_dso_trend` | `(months?) → [{month, avg_dso}]` | Are payments getting slower over time? |
| `get_break_even_estimate` | `() → {fixed_costs, variable_rate, break_even_revenue}` | What do I have to hit to stay safe? |
| `detect_anomaly` | `(metric, period?) → {anomalies: [...]}` | What drove Q3 spike? |

---

### 2.5 Accounting domain

Requires Expenses in DB + RAG corpus (Danish VAT rules). Build after both.

| Tool | Signature | Addresses |
|------|-----------|-----------|
| `get_vat_summary` | `(quarter, year) → {output_vat, input_vat, net_vat_payable}` | Danish VAT periods / audit readiness |
| `get_unreconciled_transactions` | `(days_back?) → [{transaction, amount, date}]` | What's unreconciled? |
| `get_audit_readiness_score` | `() → {score, missing_docs: [...], recommendations: [...]}` | Completeness check |
| `get_period_summary` | `(year, quarter?) → {revenue, expenses, profit, vat_position}` | P&L for accountant handoff |
| `generate_handoff_doc` | `(year, quarter?) → {markdown_summary, missing_items: [...]}` | Handoff doc generation |

---

## 3. AssistantResponse schema additions

Current schema covers action outputs but is thin on analytics. Needed before
Insights / Banking UX can work:

| Field | Type | Purpose |
|---|---|---|
| `chart_data` | `ChartData \| None` | Structured series data for frontend chart rendering |
| `metric_cards` | `list[MetricCard] \| None` | KPI tiles: net margin, runway, DSO |
| `alert` | `Alert \| None` | Proactive warning (e.g. "3 invoices 14+ days overdue") |

```python
class ChartData(BaseModel):
    chart_type: Literal["bar", "line", "pie", "area"]
    title: str
    labels: list[str]
    series: list[dict]  # [{name, data: [float]}]

class MetricCard(BaseModel):
    label: str
    value: str                                              # "42%" or "DKK 123,456"
    trend: str | None                                       # "+12% vs last period"
    sentiment: Literal["positive", "neutral", "negative"] | None

class Alert(BaseModel):
    severity: Literal["info", "warning", "critical"]
    message: str
    action_label: str | None
    action_route: str | None
```

---

## 4. Architecture improvements (from rag_poc comparison)

### 4.1 AgentRuntime Protocol
Both gateways share the same API contract but have no formal interface. Define one:

```python
# shared/runtime_protocol.py
class AgentRuntime(Protocol):
    async def run(self, input: AgentInput) -> AgentOutput: ...
    def stream(self, input: AgentInput) -> AsyncIterator[StreamEvent]: ...
    async def resume(self, thread_id: str, value: object) -> AgentOutput: ...
```

One FastAPI app could then hot-swap backends via `VA_BACKEND` env var.

### 4.2 LangGraph — injected checkpointer
Current `va-langgraph` hardcodes `MemorySaver()`. Port rag_poc pattern:

```python
def build_graph(checkpointer) -> CompiledGraph: ...
test_graph = build_graph(MemorySaver())    # test / CLI
```

Swap in `AsyncSqliteSaver` (local dev) or `AsyncPostgresSaver` (prod) in the
FastAPI lifespan without touching graph logic.

### 4.3 Multi-provider model factory
Currently hardcoded Gemini. Port rag_poc's `_resolve_model()`:

```python
# shared/model_factory.py
def resolve_llm(size: Literal["small", "medium", "large"]) -> BaseChatModel: ...
```

Reads `LLM_PROVIDER` (gemini | anthropic | openai). Lets CI/evals swap to cheaper
models without code changes.

### 4.4 ADK 1.30 compatibility
`EventsCompactionConfig` was removed in ADK 1.30. `va-google-adk/agents/va_assistant/app.py`
must switch to `before_model_callback` history pruning (rag_poc pattern) if ADK ≥1.30
is pinned. Check `pyproject.toml` before acting.

### 4.5 HITL — interrupt before destructive ops (LangGraph only)
Add `interrupt_before` on the email, void, and invite nodes. Wire gateway
`POST /chat/resume` to `Command(resume=user_response)`.

Destructive ops that warrant confirmation:
- `send_invoice_by_email` / `send_quote_by_email` (external send)
- `void_invoice` (irreversible)
- `invite_user` (external invite)

---

## 5. Build order (revised)

Work always flows: **MCP server first → agents consume via MCP**.

```
Phase 1 — Wire va agents to MCP server  ← was "Phase 7", already built
  ├── va-langgraph: replace shared/tools imports with MultiServerMCPClient
  │   Pattern: native_skill_mcp/tools.py → build_mcp_client() + load_all_billy_tools()
  │   BILLY_MCP_URL=http://127.0.0.1:8765/sse  (or stdio subprocess)
  ├── va-google-adk: swap sub-agent tools for MCPToolset
  │   Pattern: MCPToolset(StdioServerParameters(command="python",
  │              args=["-m", "app.main_noauth"], cwd="mcp_servers/billy"))
  └── Both: delete shared/tools/*.py (replaced by MCP); keep shared/schema.py

Phase 2 — Complete MCP server coverage  (no new domains, no agent changes)
  ├── Register all 6 unregistered insight tools in common.py
  │   (get_insight_revenue_summary, get_insight_invoice_status,
  │    get_insight_aging_report, get_insight_customer_summary,
  │    get_insight_product_revenue, get_invoice_lines_summary)
  ├── Add get_customer, get_product to MCP (customers.py / products.py)
  ├── Add send_invoice_reminder, get_invoice_dso_stats, void_invoice to invoices.py
  ├── Add quotes domain to MCP server
  │   (quotes table in db.py + tools/quotes.py + register in common.py)
  └── Add insights / invoice sub-agents for the 6 newly-registered tools

Phase 3 — Architecture hardening  (no domain changes)
  ├── AgentRuntime Protocol (shared/runtime_protocol.py)
  ├── Injected checkpointer (va-langgraph build_graph refactor)
  ├── Multi-provider model factory (shared/model_factory.py)
  ├── ADK 1.30 compatibility check
  └── AssistantResponse: add chart_data + metric_cards + alert fields

Phase 4 — Expenses domain  ← CRITICAL BLOCKER (unlocks Phases 5–7)
  ├── MCP server: expenses table in db.py + tools/expenses.py + register in common.py
  ├── va-google-adk: expenses_agent sub-agent
  └── va-langgraph: expense_subgraph node

Phase 5 — Banking domain  (depends on Phase 4 for burn rate)
  ├── MCP server: banking table + tools/banking.py + register
  ├── va-google-adk: banking_agent sub-agent
  └── va-langgraph: banking_subgraph node

Phase 6 — Cross-domain Insights  (depends on Phases 4+5 in DB)
  ├── MCP server: add cross-domain tools to insights section of invoices.py
  │   (get_net_margin, get_margin_by_product, get_customer_concentration,
  │    get_dso_trend, get_break_even_estimate, detect_anomaly)
  ├── va-google-adk: insights_agent sub-agent
  ├── va-langgraph: insights_subgraph + confidence routing for multi-domain queries
  └── AssistantResponse: confirm chart_data + metric_cards are wired end-to-end

Phase 7 — Accounting domain  (depends on Phase 4 + RAG corpus)
  ├── MCP server: accounting tools + VAT table
  ├── RAG corpus: Danish VAT rules, audit checklist, reporting periods
  ├── va-google-adk: accounting_agent sub-agent
  └── va-langgraph: accounting_subgraph node

Production path  (separate track, not gating dev phases)
  ├── BILLY_BACKEND=api → direct httpx wrapper around real Billy REST API
  ├── OAuth token injection from session state
  ├── Organisation ID per-request
  └── Same tool signatures — agent code unchanged
```

---

## 6. Discovery question coverage after each phase

| After phase | New coverage | Still gaps |
|-------------|-------------|------------|
| 0 (today) | Revenue mostly, DSO partial | Profitability, Break-even, Costs, Banking |
| +Phase 1 (MCP wired) | Persistent state mutations; insight panels live via MCP | Same domain gaps |
| +Phase 2 (MCP complete) | DSO analytics, AR aging, customer summary, product revenue, quote pipeline | Profitability, Break-even, Costs, Banking |
| +Phase 3 (hardening) | Architecture only — no new coverage | Same |
| +Phase 4 (Expenses) | Profitability cluster, Break-even, most of Costs | Banking, cross-domain margin |
| +Phase 5 (Banking) | Runway, cashflow forecast, balance | Cross-domain Insights |
| +Phase 6 (Insights) | Net margin, anomaly detection, customer concentration, break-even | Accounting/VAT |
| +Phase 7 (Accounting) | Audit readiness, VAT periods, handoff doc | Real Billy data (Production path) |

---

## 7. Open questions

| Phase | Question | Why it blocks |
|-------|----------|---------------|
| Phase 1 | Which ADK version is pinned? `EventsCompactionConfig` removed in 1.30 | Affects ADK MCPToolset wiring approach |
| Phase 1 | Does va-google-adk use one MCPToolset per sub-agent or shared at root? | Affects token count and session scope |
| Phase 2 | Should `send_invoice_reminder` be a new MCP tool or reuse `send_invoice_by_email` with a reminder flag? | Affects DB schema (reminder log?) |
| Phase 4 | How is `fixed_or_variable` classified for expenses? User-set flag vs LLM inference | Affects `get_expenses_by_category` and break-even calc |
| Phase 5 | Is bank balance from Billy's own bank reconciliation view, or a separate open-banking connector? | Determines whether Banking is in the Billy MCP stub or a new server |
| Phase 6 | Data quality threshold before suppressing a metric? | Avoids misleading partial metrics (e.g. margin with 2 months of expenses) |
| Prod path | Shared `tools/` package vs copy-per-project? | Currently the MCP approach makes this moot for dev; matters for prod httpx layer |

---

## 8. What this does NOT cover

- **Snowflake feature store** — cross-org analytics; separate data pipeline
- **Price recommender** — needs market benchmark data outside Billy
- **Recurring invoice automation** — Act capability requiring event triggers
- **Web client** — frontend; separate initiative
- **LangFuse observability** — separate infra initiative
- **Eval suites** — separate quality initiative

---

## Appendix: full tool inventory after all phases

### MCP server (`mcp_servers/billy/app/tools/`)

```
invoices.py     get_invoice, list_invoices, get_invoice_summary, create_invoice,
                edit_invoice, void_invoice, send_invoice_reminder, get_invoice_dso_stats,
                get_invoice_lines_summary,
                get_insight_revenue_summary, get_insight_invoice_status,
                get_insight_monthly_revenue, get_insight_top_customers,
                get_insight_aging_report, get_insight_customer_summary,
                get_insight_product_revenue                              [17]

quotes.py       list_quotes, create_quote, edit_quote,
                create_invoice_from_quote, send_quote_by_email,
                get_quote_conversion_stats                               [6]

customers.py    list_customers, create_customer, edit_customer,
                get_customer                                             [4]

products.py     list_products, create_product, edit_product,
                get_product                                              [4]

emails.py       send_invoice_by_email                                    [1]
invitations.py  invite_user                                              [1]
support_knowledge.py  fetch_support_knowledge                           [1]

expenses.py     list_expenses, get_expense, create_expense,
                get_expense_summary, get_vendor_spend,
                get_expenses_by_category, get_gross_margin              [7]

banking.py      get_bank_balance, list_bank_transactions,
                match_transaction_to_invoice, get_cashflow_forecast,
                get_runway_estimate                                      [5]

insights.py     get_net_margin, get_margin_by_product,
                get_customer_concentration, get_dso_trend,
                get_break_even_estimate, detect_anomaly                 [6]

accounting.py   get_vat_summary, get_unreconciled_transactions,
                get_audit_readiness_score, get_period_summary,
                generate_handoff_doc                                     [5]

Total: 57 tools across 11 domain files
```

### Agent domains (mirror MCP tool groupings)

ADK: one sub-agent per domain (11 sub-agents after all phases)
LangGraph: one subgraph node per domain (11 domain nodes after all phases)
Both: `shared/schema.py` (AssistantResponse) — no tool code, only output contract
