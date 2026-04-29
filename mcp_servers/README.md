# MCP Servers

Two shared MCP servers exposing accounting APIs — one stub (Billy) for dev/testing, one live (Clara/sevdesk).

## Billy — Stub Server

A shared server that exposes stub implementations of the Billy accounting API. Any agent in this repo can call tools without touching production data.

**Directory:** `billy/`

### Running

```bash
cd mcp_servers/billy && uv sync

# MCP server (STDIO mode for Claude Desktop or ADK)
python -m app.main_noauth

# MCP server (HTTP/SSE mode for remote agents)
python -m app.main_noauth --http
# Listening at http://127.0.0.1:8765/sse

# REST API
python -m app.main
# Docs at http://127.0.0.1:8766/docs
```

Or from repo root:
```bash
./scripts/run_mcp_billy.sh
./scripts/run_billy_api.sh
./scripts/reset_billy_db.sh
```

### Tools & Endpoints

Shared tool surface for both Billy and Clara (accounting, invoices, products, customers, insights):

| Tool | REST endpoint |
| --- | --- |
| `list_customers` | `GET /customers` |
| `create_customer` | `POST /customers` |
| `edit_customer` | `PATCH /customers/{id}` |
| `list_invoices` | `GET /invoices` |
| `get_invoice` | `GET /invoices/{id}` |
| `get_invoice_summary` | `GET /invoices/summary` |
| `get_invoice_lines_summary` | `GET /invoices/lines/summary` |
| `create_invoice` | `POST /invoices` |
| `edit_invoice` | `PATCH /invoices/{id}` |
| `send_invoice_by_email` | `POST /invoices/{id}/send` |
| `list_products` | `GET /products` |
| `create_product` | `POST /products` |
| `edit_product` | `PATCH /products/{id}` |
| `invite_user` | `POST /invitations` |
| `fetch_support_knowledge` | `POST /support/search` |

### Database

SQLite (`billy.db`) created on first run, auto-seeded with mock data (3 customers, 5 products, 3 invoices).

Reset to full mock dataset:
```bash
./scripts/reset_billy_db.sh
```

### Tests

```bash
cd mcp_servers/billy && uv run pytest
```

110 tests covering all tools and REST endpoints. Each test gets a clean seeded database.

---

## Clara — sevdesk MCP Server

Live sevdesk API wrapper. Mirrors Billy's tool surface against the real sevdesk API.

**Directory:** `clara/`

### Setup

```bash
cd mcp_servers/clara && uv sync
cp .env.example .env
# Edit .env with your sevdesk API key
```

### Running

```bash
# MCP server (STDIO mode)
clara

# Or directly
python -m app.main_noauth
```

### Tools

Same tool surface as Billy, backed by live sevdesk data.

### Tests

```bash
cd mcp_servers/clara && uv run pytest
```

---

## Further Reading

- [Billy SPEC.md](billy/SPEC.md) — full Billy spec: schema, database, REST API
