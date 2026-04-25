variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for the DB subnet group (must span ≥2 AZs)"
  type        = list(string)
}

variable "ecs_sg_id" {
  description = "ECS tasks security group — granted inbound access on 5432"
  type        = string
}

variable "instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_name" {
  description = "Initial database name"
  type        = string
  default     = "va_checkpoints"
}

variable "db_username" {
  description = "Postgres master username"
  type        = string
  default     = "va"
}

variable "db_password" {
  description = "Postgres master password"
  type        = string
  sensitive   = true
}
