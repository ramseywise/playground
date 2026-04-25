resource "aws_secretsmanager_secret" "gateway_api_key" {
  name                    = "${var.name_prefix}/gateway-api-key"
  description             = "X-API-Key header value for VA gateway auth"
  recovery_window_in_days = 0

  tags = { Name = "${var.name_prefix}-gateway-api-key" }
}

resource "aws_secretsmanager_secret_version" "gateway_api_key" {
  secret_id     = aws_secretsmanager_secret.gateway_api_key.id
  secret_string = var.gateway_api_key
}

resource "aws_secretsmanager_secret" "google_api_key" {
  name                    = "${var.name_prefix}/google-api-key"
  description             = "Google / Gemini API key for VA gateway containers"
  recovery_window_in_days = 0

  tags = { Name = "${var.name_prefix}-google-api-key" }
}

resource "aws_secretsmanager_secret_version" "google_api_key" {
  secret_id     = aws_secretsmanager_secret.google_api_key.id
  secret_string = var.google_api_key
}
