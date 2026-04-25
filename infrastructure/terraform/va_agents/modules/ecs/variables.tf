variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "adk_cpu" {
  description = "Fargate task CPU units for the ADK task (gateway + billy-mcp sidecar)"
  type        = number
}

variable "adk_memory" {
  description = "Fargate task memory in MiB for the ADK task"
  type        = number
}

variable "lg_cpu" {
  description = "Fargate task CPU units for the LG task (gateway + billy-mcp sidecar)"
  type        = number
}

variable "lg_memory" {
  description = "Fargate task memory in MiB for the LG task"
  type        = number
}

variable "desired_count" {
  description = "Number of ECS tasks per gateway service"
  type        = number
}

variable "adk_image_uri" {
  description = "Full ECR image URI for va-gateway-adk"
  type        = string
}

variable "lg_image_uri" {
  description = "Full ECR image URI for va-gateway-lg"
  type        = string
}

variable "billy_mcp_image_uri" {
  description = "Full ECR image URI for billy-mcp sidecar"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for ECS tasks"
  type        = list(string)
}

variable "ecs_sg_id" {
  description = "ECS tasks security group ID"
  type        = string
}

variable "adk_target_group_arn" {
  description = "ALB target group ARN for the ADK gateway"
  type        = string
}

variable "lg_target_group_arn" {
  description = "ALB target group ARN for the LG gateway"
  type        = string
}

variable "adk_listener_arn" {
  description = "ALB listener ARN for ADK — used for implicit dependency ordering"
  type        = string
}

variable "lg_listener_arn" {
  description = "ALB listener ARN for LG — used for implicit dependency ordering"
  type        = string
}

variable "gateway_api_key_arn" {
  description = "Secrets Manager ARN for GATEWAY_API_KEY"
  type        = string
}

variable "google_api_key_arn" {
  description = "Secrets Manager ARN for GOOGLE_API_KEY"
  type        = string
}

variable "gemini_model" {
  description = "Gemini model identifier passed to gateway containers"
  type        = string
  default     = "gemini-2.5-flash-lite"
}

variable "adk_log_group_name" {
  description = "CloudWatch log group for the ADK task"
  type        = string
}

variable "lg_log_group_name" {
  description = "CloudWatch log group for the LG task"
  type        = string
}

variable "billy_log_group_name" {
  description = "CloudWatch log group for billy-mcp sidecars"
  type        = string
}

variable "efs_id" {
  description = "EFS file system ID for Billy SQLite persistence"
  type        = string
}

variable "efs_access_point_id" {
  description = "EFS access point ID for /billy-data"
  type        = string
}

variable "postgres_url_secret_arn" {
  description = "Secrets Manager ARN for POSTGRES_URL (LangGraph checkpointer)"
  type        = string
}
