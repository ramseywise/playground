output "anthropic_api_key_arn" {
  description = "Secrets Manager ARN for the Anthropic API key"
  value       = aws_secretsmanager_secret.anthropic_api_key.arn
}

output "checkpoint_postgres_url_arn" {
  description = "Secrets Manager ARN for the Postgres checkpoint URL"
  value       = aws_secretsmanager_secret.checkpoint_postgres_url.arn
}
