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
# Lambda (opt-in)
# ---------------------------------------------------------------------------

variable "enable_lambda" {
  description = "Create Lambda function resources"
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

variable "lambda_auth_type" {
  description = "Lambda Function URL authorization type. Use AWS_IAM for non-dev environments."
  type        = string
  default     = "NONE"

  validation {
    condition     = contains(["NONE", "AWS_IAM"], var.lambda_auth_type)
    error_message = "lambda_auth_type must be NONE or AWS_IAM."
  }
}

variable "image_tag" {
  description = "Container image tag for Lambda (e.g. git SHA)"
  type        = string
  default     = "latest"
}

# ---------------------------------------------------------------------------
# Cross-stack (librarian_api) — required when enable_lambda = true
# ---------------------------------------------------------------------------

variable "anthropic_api_key_arn" {
  description = "Secrets Manager ARN for the Anthropic API key (from librarian_api outputs)"
  type        = string
  default     = ""
}
