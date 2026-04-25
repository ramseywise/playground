variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

# ---------------------------------------------------------------------------
# Alarm inputs — only needed after alb/ecs modules are applied
# ---------------------------------------------------------------------------

variable "alb_arn_suffix" {
  description = "ALB ARN suffix for alarm dimensions"
  type        = string
}

variable "target_group_arn_suffix" {
  description = "Target group ARN suffix for alarm dimensions"
  type        = string
}

variable "ecs_cluster_name" {
  description = "ECS cluster name for alarm dimensions"
  type        = string
}

variable "ecs_service_name" {
  description = "ECS service name for alarm dimensions"
  type        = string
}

variable "alarm_sns_arn" {
  description = "SNS topic ARN for alarm notifications. Empty string disables notifications."
  type        = string
  default     = ""
}
