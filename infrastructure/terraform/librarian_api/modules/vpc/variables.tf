variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
}

variable "container_port" {
  description = "Port the ECS container listens on (used for ECS security group ingress)"
  type        = number
}
