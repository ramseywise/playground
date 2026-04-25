variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "eu-west-1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "librarian"
}

# ---------------------------------------------------------------------------
# ECS / Fargate
# ---------------------------------------------------------------------------

variable "cpu" {
  description = "Fargate task CPU units (256, 512, 1024, 2048, 4096)"
  type        = number
  default     = 512
}

variable "memory" {
  description = "Fargate task memory in MiB (min 4096 for multilingual-e5-large)"
  type        = number
  default     = 4096
}

variable "desired_count" {
  description = "Number of ECS tasks to run"
  type        = number
  default     = 1
}

variable "container_port" {
  description = "Port the container listens on"
  type        = number
  default     = 8000
}

variable "image_tag" {
  description = "Container image tag to deploy (e.g. git SHA). Defaults to 'latest' for local dev only."
  type        = string
  default     = "latest"
}

# ---------------------------------------------------------------------------
# Secrets (passed via .tfvars or -var, never committed)
# ---------------------------------------------------------------------------

variable "anthropic_api_key" {
  description = "Anthropic API key — stored in Secrets Manager"
  type        = string
  sensitive   = true
}

variable "checkpoint_postgres_url" {
  description = "Postgres connection URL for LangGraph checkpointer — stored in Secrets Manager"
  type        = string
  sensitive   = true
  default     = ""
}

# ---------------------------------------------------------------------------
# HTTPS (optional)
# ---------------------------------------------------------------------------

variable "enable_https" {
  description = "Enable HTTPS listener and HTTP→HTTPS redirect. Requires acm_cert_arn."
  type        = bool
  default     = false
}

variable "acm_cert_arn" {
  description = "ACM certificate ARN for the HTTPS listener. Required when enable_https = true."
  type        = string
  default     = ""
}

# ---------------------------------------------------------------------------
# Cross-stack (librarian_llm) — set after llm stack is deployed
# ---------------------------------------------------------------------------

variable "s3_bucket_arn" {
  description = "S3 data lake bucket ARN from librarian_llm stack. Empty = no S3 policy on ECS task role."
  type        = string
  default     = ""
}

variable "alarm_sns_arn" {
  description = "SNS topic ARN for CloudWatch alarm notifications. Empty = alarms fire silently."
  type        = string
  default     = ""
}

# ---------------------------------------------------------------------------
# Snowflake (optional, for MCP server)
# ---------------------------------------------------------------------------

variable "snowflake_account" {
  description = "Snowflake account identifier"
  type        = string
  default     = ""
}

variable "snowflake_user" {
  description = "Snowflake username"
  type        = string
  default     = ""
}

variable "snowflake_password" {
  description = "Snowflake password"
  type        = string
  sensitive   = true
  default     = ""
}
