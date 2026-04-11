output "alb_dns_name" {
  description = "Public DNS name of the ALB"
  value       = aws_lb.api.dns_name
}

output "ecr_repository_url" {
  description = "ECR repository URL for docker push"
  value       = aws_ecr_repository.api.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.api.name
}

output "lambda_function_url" {
  description = "Lambda function URL (when enabled)"
  value       = var.enable_lambda ? aws_lambda_function_url.api[0].function_url : ""
  sensitive   = true
}

output "s3_bucket_name" {
  description = "S3 data lake bucket name"
  value       = aws_s3_bucket.data_lake.id
}
