# Infrastructure

Container definitions and Terraform stacks for two platforms: **Librarian** and **VA Agents**.

## Layout

```
infrastructure/
  containers/
    docker-compose.librarian.yml   # Local dev: Librarian api + frontend + eval-dashboard
    docker-compose.va.yml          # Local dev: VA gateways + Billy MCP
    librarian/
      api/Dockerfile
      frontend/Dockerfile
      eval-dashboard/Dockerfile
    va-gateway-adk/Dockerfile      # FastAPI + Google ADK runner
    va-gateway-lg/Dockerfile       # FastAPI + LangGraph runner
    billy-mcp/
      Dockerfile                   # Billy MCP SSE (:8765) + REST API (:8766)
      entrypoint.sh
  terraform/
    _shared/                       # Cross-stack data sources and outputs
    librarian_api/                 # ECS stack for Librarian API
    librarian_llm/                 # Lambda stack for LLM ingestion/inference
    va_agents/                     # (planned — Track B Phase 2) ECS stack for VA agents
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

### Terraform (`va_agents`) — planned

Track B Phase 2 will add `terraform/va_agents/` with:
- ECS Fargate tasks for both gateways + Billy MCP sidecar
- ALB with path-based routing and SSE-compatible idle timeout (300s)
- ECR repos for all three images
- Secrets Manager entries for `GATEWAY_API_KEY`, `GOOGLE_API_KEY`, `BILLY_DB`
- RDS Postgres module (Phase 3)

---

## Librarian

### Local dev

```sh
cd infrastructure/containers
docker compose -f docker-compose.librarian.yml up
```

Services: `api` on :8000, `frontend` on :3000, `eval-dashboard` on :8080.

### Stacks

#### `librarian_api`
ECS Fargate service behind an ALB. Manages its own VPC, subnets, security groups, ECR repo, Secrets Manager entries, and CloudWatch alarms.

```sh
cd terraform/librarian_api
terraform init
terraform apply -var-file=environments/dev.tfvars
```

#### `librarian_llm`
Lambda functions for LLM ingestion and inference. Reads S3 for input, writes results back. Depends on `librarian_api` remote state.

```sh
cd terraform/librarian_llm
terraform init
terraform apply -var-file=environments/dev.tfvars
```

Apply `librarian_api` before `librarian_llm` — the LLM stack reads its remote state.

Each stack has `environments/dev.tfvars` and `environments/prod.tfvars`. Backend config is in `backend.tf` per stack.
