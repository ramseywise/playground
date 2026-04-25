output "lambda_exec_role_arn" {
  description = "Lambda execution role ARN"
  value       = var.enable_lambda ? aws_iam_role.lambda_exec[0].arn : ""
}

output "lambda_exec_role_name" {
  description = "Lambda execution role name"
  value       = var.enable_lambda ? aws_iam_role.lambda_exec[0].name : ""
}
