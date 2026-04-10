# ---------------------------------------------------------------------------
# Lambda — container image deployment (opt-in via enable_lambda)
# ---------------------------------------------------------------------------

# IAM role for Lambda execution
resource "aws_iam_role" "lambda_exec" {
  count = var.enable_lambda ? 1 : 0

  name               = "${local.name_prefix}-lambda-exec"
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

# Lambda needs to read Secrets Manager (same secret as ECS)
resource "aws_iam_role_policy" "lambda_secrets" {
  count = var.enable_lambda ? 1 : 0

  name   = "${local.name_prefix}-lambda-secrets"
  role   = aws_iam_role.lambda_exec[0].id
  policy = data.aws_iam_policy_document.secrets_read.json
}

# Lambda needs S3 read/write for data lake
resource "aws_iam_role_policy" "lambda_s3" {
  count = var.enable_lambda ? 1 : 0

  name   = "${local.name_prefix}-lambda-s3"
  role   = aws_iam_role.lambda_exec[0].id
  policy = data.aws_iam_policy_document.s3_readwrite.json
}

# API Lambda — same container image, Mangum handler
resource "aws_lambda_function" "api" {
  count = var.enable_lambda ? 1 : 0

  function_name = "${local.name_prefix}-api"
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api.repository_url}:latest"
  role          = aws_iam_role.lambda_exec[0].arn
  memory_size   = var.lambda_memory
  timeout       = var.lambda_timeout

  image_config {
    command = ["agents.librarian.api.lambda_handler.handler"]
  }

  environment {
    variables = {
      RETRIEVAL_STRATEGY   = "chroma"
      RERANKER_STRATEGY    = "cross_encoder"
      CONFIDENCE_THRESHOLD = "0.4"
      LAMBDA_EXECUTION     = "true"
      DUCKDB_PATH          = "/tmp/librarian.db"
    }
  }

  tags = { Name = "${local.name_prefix}-api-lambda" }
}

# Function URL — direct HTTP access without API Gateway
resource "aws_lambda_function_url" "api" {
  count = var.enable_lambda ? 1 : 0

  function_name      = aws_lambda_function.api[0].function_name
  authorization_type = "NONE"
}

# Ingestion Lambda — triggered by S3 events on raw/ prefix
resource "aws_lambda_function" "ingestion" {
  count = var.enable_lambda ? 1 : 0

  function_name = "${local.name_prefix}-ingestion"
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api.repository_url}:latest"
  role          = aws_iam_role.lambda_exec[0].arn
  memory_size   = var.lambda_memory
  timeout       = var.lambda_timeout

  image_config {
    command = ["agents.librarian.api.s3_trigger.handler"]
  }

  environment {
    variables = {
      LAMBDA_EXECUTION   = "true"
      RETRIEVAL_STRATEGY = "chroma"
      DUCKDB_PATH        = "/tmp/librarian.db"
    }
  }

  tags = { Name = "${local.name_prefix}-ingestion-lambda" }
}

# Allow S3 to invoke the ingestion Lambda
resource "aws_lambda_permission" "s3_invoke_ingestion" {
  count = var.enable_lambda ? 1 : 0

  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingestion[0].function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.data_lake.arn
}
