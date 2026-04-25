variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for EFS mount targets (one per AZ)"
  type        = list(string)
}

variable "ecs_sg_id" {
  description = "ECS tasks security group — granted NFS inbound on 2049"
  type        = string
}
