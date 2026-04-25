variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
}

variable "enable_lambda" {
  description = "Create Lambda function resources"
  type        = bool
  default     = false
}

variable "image_uri" {
  description = "Full ECR image URI including tag for Lambda container image"
  type        = string
  default     = ""
}

variable "lambda_memory" {
  description = "Lambda function memory in MB"
  type        = number
  default     = 1024
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 60
}

variable "lambda_auth_type" {
  description = "Lambda Function URL authorization type. Use AWS_IAM for non-dev environments."
  type        = string
  default     = "NONE"

  validation {
    condition     = contains(["NONE", "AWS_IAM"], var.lambda_auth_type)
    error_message = "lambda_auth_type must be NONE or AWS_IAM."
  }
}

variable "lambda_exec_role_arn" {
  description = "IAM role ARN for Lambda execution (from iam module)"
  type        = string
  default     = ""
}

variable "s3_bucket_arn" {
  description = "S3 data lake bucket ARN (for Lambda permission and S3 notification)"
  type        = string
  default     = ""
}

variable "s3_bucket_id" {
  description = "S3 data lake bucket ID (name) for S3 event notification"
  type        = string
  default     = ""
}
