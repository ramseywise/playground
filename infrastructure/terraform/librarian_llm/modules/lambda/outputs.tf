output "function_url" {
  description = "Lambda function URL (when enabled)"
  value       = var.enable_lambda ? aws_lambda_function_url.api[0].function_url : ""
  sensitive   = true
}
