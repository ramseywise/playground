output "ecr_repository_url" {
  description = "ECR repository URL from librarian_api stack — use as Lambda image URI base"
  value       = data.terraform_remote_state.librarian_api.outputs.ecr_repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name from librarian_api stack"
  value       = data.terraform_remote_state.librarian_api.outputs.ecs_cluster_name
}
