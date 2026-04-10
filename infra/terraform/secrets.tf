# ---------------------------------------------------------------------------
# Secrets Manager — API keys injected into ECS tasks as env vars
# ---------------------------------------------------------------------------

resource "aws_secretsmanager_secret" "anthropic_api_key" {
  name                    = "${local.name_prefix}/anthropic-api-key"
  recovery_window_in_days = 0 # dev convenience — set to 7+ for prod
  tags                    = { Name = "${local.name_prefix}-anthropic-key" }
}

resource "aws_secretsmanager_secret_version" "anthropic_api_key" {
  secret_id     = aws_secretsmanager_secret.anthropic_api_key.id
  secret_string = var.anthropic_api_key
}
