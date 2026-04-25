variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "cpu" {
  description = "Fargate task CPU units (256, 512, 1024, 2048, 4096)"
  type        = number
}

variable "memory" {
  description = "Fargate task memory in MiB"
  type        = number
}

variable "desired_count" {
  description = "Number of ECS tasks to run"
  type        = number
}

variable "container_port" {
  description = "Port the container listens on"
  type        = number
}

variable "image_uri" {
  description = "Full ECR image URI including tag (e.g. 123456789.dkr.ecr.eu-west-1.amazonaws.com/librarian-dev-api:abc1234)"
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

variable "target_group_arn" {
  description = "ALB target group ARN to register tasks with"
  type        = string
}

variable "alb_listener_arn" {
  description = "ALB HTTP listener ARN — passed to create implicit dependency ordering"
  type        = string
}

variable "anthropic_api_key_arn" {
  description = "Secrets Manager ARN for the Anthropic API key"
  type        = string
}

variable "log_group_name" {
  description = "CloudWatch log group name for container logs"
  type        = string
}

variable "s3_bucket_arn" {
  description = "S3 data lake bucket ARN for ECS task role. Set after librarian_llm stack is deployed."
  type        = string
  default     = ""
}
