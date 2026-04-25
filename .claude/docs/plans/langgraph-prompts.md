# Plan: LangGraph Prompt Extraction & Fleshing Out

## Goal

Extract all inline prompt strings from `va-langgraph` Python files into `prompts/` text files, then flesh out the domain prompts to match the detail level of the ADK equivalents.

## Context

LangGraph uses a two-stage architecture:
- **Domain subgraph** (`domains.py`): calls MCP tools, collects results into `state.tool_results`
- **`format_node`** (`nodes/format.py`): takes `tool_results` + user message → renders `AssistantResponse`

Domain prompts do NOT need response format guidance — that lives in `format_node`'s prompt. They only need tool-selection and domain-rule guidance.

ADK prompts are the reference for content. Exclude the `## Response Format` section from ADK (LangGraph doesn't need it per-domain).

## Files with inline prompts

| File | Variable | Target file |
|------|----------|-------------|
| `graph/nodes/analyze.py` | `_SYSTEM` | `prompts/router.txt` |
| `graph/nodes/format.py` | `_SYSTEM` | `prompts/format.txt` |
| `graph/nodes/direct.py` | `_SYSTEM` | `prompts/direct.txt` |
| `graph/nodes/memory.py` | `_EXTRACT_SYSTEM` | `prompts/memory_extract.txt` |
| `graph/subgraphs/domains.py` | `_INVOICE_SYSTEM` | `prompts/invoice.txt` |
| `graph/subgraphs/domains.py` | `_QUOTE_SYSTEM` | `prompts/quote.txt` |
| `graph/subgraphs/domains.py` | `_CUSTOMER_SYSTEM` | `prompts/customer.txt` |
| `graph/subgraphs/domains.py` | `_PRODUCT_SYSTEM` | `prompts/product.txt` |
| `graph/subgraphs/domains.py` | `_EMAIL_SYSTEM` | `prompts/email.txt` |
| `graph/subgraphs/domains.py` | `_INVITATION_SYSTEM` | `prompts/invitation.txt` |
| `graph/subgraphs/domains.py` | `_EXPENSE_SYSTEM` | `prompts/expense.txt` |
| `graph/subgraphs/domains.py` | `_BANKING_SYSTEM` | `prompts/banking.txt` |
| `graph/subgraphs/domains.py` | `_ACCOUNTING_SYSTEM` | `prompts/accounting.txt` |
| `graph/subgraphs/domains.py` | `_SUPPORT_SYSTEM` | `prompts/support.txt` |
| `graph/subgraphs/domains.py` | `_INSIGHTS_SYSTEM` | `prompts/insights.txt` |

## Steps

- [ ] 1. Create `va-langgraph/prompts/` directory
- [ ] 2. Write all 15 prompt files (extract current content + flesh out with ADK-level detail)
- [ ] 3. Update `graph/nodes/analyze.py` to load `router.txt` from file
- [ ] 4. Update `graph/nodes/format.py` to load `format.txt` from file
- [ ] 5. Update `graph/nodes/direct.py` to load `direct.txt` from file
- [ ] 6. Update `graph/nodes/memory.py` to load `memory_extract.txt` from file
- [ ] 7. Update `graph/subgraphs/domains.py` to load all 11 domain prompts from files
- [ ] 8. Verify: `uv run pytest tests/ -v` passes

## What "fleshed out" means per domain

Domain prompts should NOT include response format (format_node handles that).
They SHOULD include:

- **Which tool to use for which user intent** (currently missing from most domains)
- **Parallel tool call guidance** where relevant
- **Domain-specific rules** (VAT 25%, DKK, Danish conventions)
- **Confirmation flow** before creating/editing (currently missing)
- **Edge cases** (e.g. can't edit approved invoice, empty tool sets pending test account)

## Reference

ADK prompts live in `va-google-adk/prompts/` — use as content reference but strip the `## Response Format` section.
