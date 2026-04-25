variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_ids" {
  description = "Public subnet IDs for the ALB"
  type        = list(string)
}

variable "alb_sg_id" {
  description = "ALB security group ID"
  type        = string
}

variable "enable_https" {
  description = "Enable HTTPS listener and HTTP→HTTPS redirect on the ADK listener."
  type        = bool
  default     = false
}

variable "acm_cert_arn" {
  description = "ACM certificate ARN. Required when enable_https = true."
  type        = string
  default     = ""
}
