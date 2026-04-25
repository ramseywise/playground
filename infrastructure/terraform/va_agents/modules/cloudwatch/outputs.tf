output "adk_log_group_name" {
  description = "CloudWatch log group name for the ADK task"
  value       = aws_cloudwatch_log_group.adk.name
}

output "lg_log_group_name" {
  description = "CloudWatch log group name for the LG task"
  value       = aws_cloudwatch_log_group.lg.name
}

output "billy_log_group_name" {
  description = "CloudWatch log group name for billy-mcp sidecars"
  value       = aws_cloudwatch_log_group.billy.name
}
