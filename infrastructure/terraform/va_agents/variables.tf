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
  default     = "va-agents"
}

variable "adk_cpu" {
  description = "Fargate task CPU units for the ADK gateway task (includes billy-mcp sidecar)"
  type        = number
  default     = 1024
}

variable "adk_memory" {
  description = "Fargate task memory in MiB for the ADK gateway task"
  type        = number
  default     = 2048
}

variable "lg_cpu" {
  description = "Fargate task CPU units for the LangGraph gateway task (includes billy-mcp sidecar)"
  type        = number
  default     = 1024
}

variable "lg_memory" {
  description = "Fargate task memory in MiB for the LangGraph gateway task"
  type        = number
  default     = 2048
}

variable "desired_count" {
  description = "Number of ECS tasks to run for each gateway service"
  type        = number
  default     = 1
}

variable "adk_image_tag" {
  description = "Image tag for va-gateway-adk (e.g. git SHA). Defaults to 'latest' for local dev only."
  type        = string
  default     = "latest"
}

variable "lg_image_tag" {
  description = "Image tag for va-gateway-lg"
  type        = string
  default     = "latest"
}

variable "billy_mcp_image_tag" {
  description = "Image tag for billy-mcp"
  type        = string
  default     = "latest"
}

variable "gateway_api_key" {
  description = "API key clients must send in X-API-Key header — stored in Secrets Manager"
  type        = string
  sensitive   = true
}

variable "google_api_key" {
  description = "Google / Gemini API key — stored in Secrets Manager"
  type        = string
  sensitive   = true
}

variable "gemini_model" {
  description = "Gemini model identifier"
  type        = string
  default     = "gemini-2.5-flash-lite"
}

variable "enable_https" {
  description = "Enable HTTPS listener and HTTP→HTTPS redirect. Requires acm_cert_arn."
  type        = bool
  default     = false
}

variable "acm_cert_arn" {
  description = "ACM certificate ARN for HTTPS. Required when enable_https = true."
  type        = string
  default     = ""
}

variable "alarm_sns_arn" {
  description = "SNS topic ARN for CloudWatch alarm notifications. Empty = alarms fire silently."
  type        = string
  default     = ""
}

variable "db_password" {
  description = "Postgres master password for the RDS instance (stored in Secrets Manager)"
  type        = string
  sensitive   = true
}
