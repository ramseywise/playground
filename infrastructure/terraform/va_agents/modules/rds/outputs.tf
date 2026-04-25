output "db_endpoint" {
  description = "RDS endpoint (host:port)"
  value       = aws_db_instance.main.endpoint
}

output "postgres_url_secret_arn" {
  description = "Secrets Manager ARN for POSTGRES_URL"
  value       = aws_secretsmanager_secret.postgres_url.arn
}
