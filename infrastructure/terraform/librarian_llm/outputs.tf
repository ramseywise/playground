output "s3_bucket_name" {
  description = "S3 data lake bucket name"
  value       = module.s3.bucket_id
}

output "s3_bucket_arn" {
  description = "S3 data lake bucket ARN — pass to librarian_api var.s3_bucket_arn"
  value       = module.s3.bucket_arn
}

output "lambda_function_url" {
  description = "Lambda function URL (when enabled)"
  value       = module.lambda.function_url
  sensitive   = true
}
