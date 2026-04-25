output "cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "adk_service_name" {
  description = "ECS service name for va-gateway-adk"
  value       = aws_ecs_service.adk.name
}

output "lg_service_name" {
  description = "ECS service name for va-gateway-lg"
  value       = aws_ecs_service.lg.name
}

output "task_role_arn" {
  description = "ECS task role ARN (shared by both services)"
  value       = aws_iam_role.task.arn
}
