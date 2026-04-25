variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
}

variable "enable_lambda" {
  description = "Create Lambda IAM resources (mirrors enable_lambda in lambda module)"
  type        = bool
  default     = false
}

variable "s3_readwrite_policy_json" {
  description = "IAM policy JSON granting read/write on the S3 data lake (output from s3 module)"
  type        = string
}

variable "anthropic_api_key_arn" {
  description = "Secrets Manager ARN for the Anthropic API key (cross-stack from librarian_api)"
  type        = string
}
