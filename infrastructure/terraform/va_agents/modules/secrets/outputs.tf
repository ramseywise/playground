output "gateway_api_key_arn" {
  description = "Secrets Manager ARN for GATEWAY_API_KEY"
  value       = aws_secretsmanager_secret.gateway_api_key.arn
}

output "google_api_key_arn" {
  description = "Secrets Manager ARN for GOOGLE_API_KEY"
  value       = aws_secretsmanager_secret.google_api_key.arn
}
