# Infrastructure

Container definitions and Terraform stacks for Billy VA (two implementations: Google ADK and LangGraph).

## Layout

```
infrastructure/
  containers/
    docker-compose.va.yml          # Local dev: VA gateways + Billy MCP
    frontend/                      # Next.js frontend
    billy-mcp/
      Dockerfile                   # Billy MCP SSE (:8765) + REST API (:8766)
      entrypoint.sh
    va-gateway-adk/Dockerfile      # FastAPI + Google ADK runner
    va-gateway-lg/Dockerfile       # FastAPI + LangGraph runner
    va-support-rag/Dockerfile      # Support knowledge RAG service
    postgres-init/                 # RDS init scripts
  terraform/
    va_agents/                     # ECS Fargate stack for both gateways + Billy MCP
```

---

## VA Agents

### Local dev

```sh
cd infrastructure/containers
docker compose -f docker-compose.va.yml up
```

Services:
- `va-gateway-adk` on :8000 — Google ADK runner with FastAPI gateway
- `va-gateway-lg` on :8001 — LangGraph runner with FastAPI gateway
- `billy-mcp` on :8765 (MCP SSE) + :8766 (REST/Swagger) — Billy tool layer

Both gateways depend on `billy-mcp` being healthy before starting.

### Phase 3: Postgres checkpointer (opt-in)

```sh
docker compose -f docker-compose.va.yml --profile postgres up
```

Adds a `postgres:16` container on :5432. Set `POSTGRES_URL` and `LANGGRAPH_CHECKPOINTER=postgres` in your `.env`.

### Terraform (`va_agents`)

```sh
cd terraform/va_agents
terraform init
terraform apply -var-file=environments/dev.tfvars
```

Includes:
- ECS Fargate tasks for both gateways + Billy MCP sidecar
- ALB with path-based routing and SSE-compatible idle timeout (300s)
- ECR repos for all three images
- Secrets Manager entries for `GATEWAY_API_KEY`, `GOOGLE_API_KEY`, `BILLY_DB`
- RDS Postgres module for checkpointing
