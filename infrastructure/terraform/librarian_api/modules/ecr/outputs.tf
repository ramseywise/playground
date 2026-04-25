output "repository_url" {
  description = "ECR repository URL for docker push"
  value       = aws_ecr_repository.api.repository_url
}
