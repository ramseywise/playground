output "gateway_adk_url" {
  description = "ECR repo URL for va-gateway-adk"
  value       = aws_ecr_repository.gateway_adk.repository_url
}

output "gateway_lg_url" {
  description = "ECR repo URL for va-gateway-lg"
  value       = aws_ecr_repository.gateway_lg.repository_url
}

output "billy_mcp_url" {
  description = "ECR repo URL for billy-mcp"
  value       = aws_ecr_repository.billy_mcp.repository_url
}
