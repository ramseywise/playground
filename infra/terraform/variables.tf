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
  description = "Fargate task memory in MiB"
  type        = number
  default     = 1024
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

# ---------------------------------------------------------------------------
# Secrets (passed via terraform.tfvars or -var, never committed)
# ---------------------------------------------------------------------------

variable "anthropic_api_key" {
  description = "Anthropic API key — stored in Secrets Manager"
  type        = string
  sensitive   = true
}

# ---------------------------------------------------------------------------
# Lambda (opt-in)
# ---------------------------------------------------------------------------

variable "enable_lambda" {
  description = "Create Lambda function resources alongside Fargate"
  type        = bool
  default     = false
}

variable "lambda_memory" {
  description = "Lambda function memory in MB"
  type        = number
  default     = 1024
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 60
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
