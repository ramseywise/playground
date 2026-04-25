output "log_group_name" {
  description = "CloudWatch log group name for ECS container logs"
  value       = aws_cloudwatch_log_group.api.name
}
