# ---------------------------------------------------------------------------
# IAM — Lambda execution role + policies
# ---------------------------------------------------------------------------

resource "aws_iam_role" "lambda_exec" {
  count = var.enable_lambda ? 1 : 0

  name = "${var.name_prefix}-lambda-exec"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  count = var.enable_lambda ? 1 : 0

  role       = aws_iam_role.lambda_exec[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Allow reading Secrets Manager (Anthropic API key from librarian_api stack)
data "aws_iam_policy_document" "secrets_read" {
  count = var.enable_lambda ? 1 : 0

  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [var.anthropic_api_key_arn]
  }
}

resource "aws_iam_role_policy" "lambda_secrets" {
  count = var.enable_lambda ? 1 : 0

  name   = "${var.name_prefix}-lambda-secrets"
  role   = aws_iam_role.lambda_exec[0].id
  policy = data.aws_iam_policy_document.secrets_read[0].json
}

# Allow read/write on S3 data lake
resource "aws_iam_role_policy" "lambda_s3" {
  count = var.enable_lambda ? 1 : 0

  name   = "${var.name_prefix}-lambda-s3"
  role   = aws_iam_role.lambda_exec[0].id
  policy = var.s3_readwrite_policy_json
}
