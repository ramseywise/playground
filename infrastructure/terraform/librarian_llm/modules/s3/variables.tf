variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
}

variable "environment" {
  description = "Deployment environment — controls force_destroy on the bucket"
  type        = string
}
