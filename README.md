# Billy VA

Virtual assistant for [Billy](https://www.billy.dk/) — implemented in two frameworks for comparison.

## Architecture

```
va-google-adk/      Google ADK — router agent + 11 domain sub-agents
va-langgraph/       LangGraph — StateGraph with domain subgraphs
mcp_servers/billy/  Shared MCP + REST backend (Billy data + support knowledge)
infrastructure/     Docker Compose (local) + Terraform (AWS)
```

Both VAs expose the same HTTP API shape (`/chat`, `/chat/stream`, `/health`) and connect to the same Billy MCP server. The ADK implementation uses Google's multi-agent framework; the LangGraph one is a handcrafted state machine. Both run Gemini 2.5 Flash.

## Quick start

```bash
# Prerequisites: Docker, a GOOGLE_API_KEY

cp va-google-adk/.env.example va-google-adk/.env   # fill in GOOGLE_API_KEY
cp va-langgraph/.env.example va-langgraph/.env

make va-up        # full stack: frontend + billy-mcp + both gateways + postgres
make va-up-ui     # UI only: frontend + billy-mcp (fastest)
make va-smoke     # health-check all running services
make va-down      # tear down
```

Services:

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | Chat UI |
| Billy MCP | http://localhost:8765/sse | MCP server (SSE) |
| Billy REST | http://localhost:8766/docs | Swagger / REST API |
| VA Gateway (ADK) | http://localhost:8000 | Google ADK agent |
| VA Gateway (LG) | http://localhost:8001 | LangGraph agent |

## Sub-project tests

```bash
cd va-google-adk && uv run pytest tests/ -v
cd va-langgraph  && uv run pytest tests/ -v
cd mcp_servers/billy && uv run pytest tests/ -v
```

## Stack

- **Google ADK** (`va-google-adk`) — multi-agent orchestration, Gemini 2.5 Flash
- **LangGraph** (`va-langgraph`) — StateGraph, Postgres checkpointing, Gemini 2.5 Flash
- **FastMCP / FastAPI** (`mcp_servers/billy`) — Billy accounting data + support knowledge
- **Terraform** (`infrastructure/terraform`) — ECS/Fargate, ALB, RDS, ECR on AWS
