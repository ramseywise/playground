variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "alb_arn_suffix" {
  description = "ALB ARN suffix for alarm dimensions"
  type        = string
}

variable "adk_target_group_arn_suffix" {
  description = "ADK target group ARN suffix for alarm dimensions"
  type        = string
}

variable "lg_target_group_arn_suffix" {
  description = "LG target group ARN suffix for alarm dimensions"
  type        = string
}

variable "ecs_cluster_name" {
  description = "ECS cluster name for alarm dimensions"
  type        = string
}

variable "ecs_adk_service_name" {
  description = "ADK ECS service name for alarm dimensions"
  type        = string
}

variable "ecs_lg_service_name" {
  description = "LG ECS service name for alarm dimensions"
  type        = string
}

variable "alarm_sns_arn" {
  description = "SNS topic ARN for alarm notifications. Empty string disables notifications."
  type        = string
  default     = ""
}
