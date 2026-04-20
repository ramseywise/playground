# CHANGELOG

---

## 2026-04-15 — terraform-restructure (Steps 1–6)

**Plan**: [terraform-restructure/plan.md](in-progress/terraform-restructure/plan.md)
**Status**: Steps 1–6 complete. Step 7 (delete old `infra/`) pending validation.

### Step 1 ✅ Move containers
- Created `infrastructure/containers/{api,frontend,eval-dashboard}/Dockerfile`
- Created `infrastructure/containers/docker-compose.yml`
- Updated `build.dockerfile` paths (context `../..` unchanged — same relative depth)
- Old `infra/docker/` preserved until Step 7 validation

### Step 2 ✅ Extract librarian_api modules
Flat `infra/terraform/*.tf` decomposed into `infrastructure/infrastructure_as_code/librarian_api/modules/`:

| Source | Module |
|---|---|
| `vpc.tf` + `security.tf` | `modules/vpc/` (SGs co-located with VPC) |
| `ecr.tf` | `modules/ecr/` |
| `alb.tf` | `modules/alb/` |
| `secrets.tf` | `modules/secrets/` |
| `ecs.tf` (ECS resources) | `modules/ecs/` |
| `ecs.tf` (CloudWatch log group) | `modules/cloudwatch/` |

`moved {}` blocks in `librarian_api/main.tf` cover all address changes for zero-diff plan after state migration.

**State migration required before `terraform plan` will be clean:**
```bash
# Copy existing state to new directory, then:
cp infra/terraform/terraform.tfstate \
   infrastructure/infrastructure_as_code/librarian_api/terraform.tfstate
cd infrastructure/infrastructure_as_code/librarian_api
terraform init -migrate-state  # after bootstrapping S3 bucket + DynamoDB table
terraform plan -var-file=environments/dev.tfvars  # should show 0 changes
```

### Step 3 ✅ Extract librarian_llm stack
S3 + Lambda + IAM resources moved to separate stack `infrastructure/infrastructure_as_code/librarian_llm/`.
Cross-stack wiring via `_shared/data.tf` (reads librarian_api S3 remote state).

**Resources requiring `terraform state mv` between state files** (run against original state):
```bash
# Example — repeat for each resource in the list below
terraform -chdir=infra/terraform state mv \
  -state-out=../../infrastructure/infrastructure_as_code/librarian_llm/terraform.tfstate \
  aws_s3_bucket.data_lake \
  module.s3.aws_s3_bucket.data_lake
```

Full resource list documented in `librarian_llm/main.tf` comments.

**Deployment order:**
1. `librarian_api` (creates ECR + Secrets Manager)
2. `librarian_llm` (reads ECR URL from `_shared`, creates S3)
3. Re-apply `librarian_api` with `s3_bucket_arn` set (attaches S3 policy to ECS task role)

### Step 4 ✅ Environment tfvars + S3 backend
- `environments/{dev,prod}.tfvars` for both stacks
- `backend.tf` configured with S3 + DynamoDB locking
- Bootstrap instructions in `librarian_api/backend.tf` comments

### Step 5 ✅ CloudWatch module (additive)
Added to `modules/cloudwatch/main.tf`:
- `aws_cloudwatch_metric_alarm.alb_5xx_rate` — 5xx rate > 5% over 5 min
- `aws_cloudwatch_metric_alarm.alb_latency_p99` — p99 > 2s over 15 min
- `aws_cloudwatch_metric_alarm.ecs_task_count` — running tasks = 0

### Step 6 ✅ GitHub Actions CI/CD — unblocks github-cicd-pipeline
- `.github/workflows/ci.yml` — lint → test → typecheck → terraform plan (PR) / build+push+deploy (main)
- `.github/workflows/deploy.yml` — reusable ECS rolling deploy with env gate + health check

**Prerequisite**: GitHub App `workflows` permission must be granted (see `backlog/github-cicd-pipeline/plan.md` Step 1).

**Repo variables to configure:**
- `AWS_GITHUB_ACTIONS_ROLE_ARN` — OIDC role for CI
- `ECR_REPOSITORY_NAME` — ECR repo name
- `staging_AWS_ROLE_ARN`, `production_AWS_ROLE_ARN` — per-env deploy roles
- GitHub environment protection rules for `staging` and `production`

### Step 7 ⏳ Delete old infra/ (pending validation)
Once `terraform plan` shows zero diff in both new stacks, run:
```bash
rm -rf infra/terraform/ infra/docker/
# Update any remaining references:
grep -r "infra/docker\|infra/terraform" . --include="*.md" --include="*.yml"
```
