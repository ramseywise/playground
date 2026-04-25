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

variable "container_port" {
  description = "Port the container listens on"
  type        = number
}

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
