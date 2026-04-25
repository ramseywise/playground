output "alb_dns_name" {
  description = "Public DNS name of the ALB (adk on :80, lg on :8001)"
  value       = module.alb.alb_dns_name
}

output "ecr_gateway_adk_url" {
  description = "ECR repo URL for va-gateway-adk"
  value       = module.ecr.gateway_adk_url
}

output "ecr_gateway_lg_url" {
  description = "ECR repo URL for va-gateway-lg"
  value       = module.ecr.gateway_lg_url
}

output "ecr_billy_mcp_url" {
  description = "ECR repo URL for billy-mcp"
  value       = module.ecr.billy_mcp_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = module.ecs.cluster_name
}

output "ecs_adk_service_name" {
  description = "ECS service name for va-gateway-adk"
  value       = module.ecs.adk_service_name
}

output "ecs_lg_service_name" {
  description = "ECS service name for va-gateway-lg"
  value       = module.ecs.lg_service_name
}
