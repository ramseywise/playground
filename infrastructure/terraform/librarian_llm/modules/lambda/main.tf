# ---------------------------------------------------------------------------
# Lambda — container image deployment (opt-in via enable_lambda)
# ---------------------------------------------------------------------------

# API Lambda — same container image as ECS, Mangum handler
resource "aws_lambda_function" "api" {
  count = var.enable_lambda ? 1 : 0

  function_name = "${var.name_prefix}-api"
  package_type  = "Image"
  image_uri     = var.image_uri
  role          = var.lambda_exec_role_arn
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

  tags = { Name = "${var.name_prefix}-api-lambda" }
}

# Function URL — direct HTTP access without API Gateway
resource "aws_lambda_function_url" "api" {
  count = var.enable_lambda ? 1 : 0

  function_name      = aws_lambda_function.api[0].function_name
  authorization_type = var.lambda_auth_type
  # SECURITY: default is NONE (dev). Set to AWS_IAM for staging/prod.
}

# Ingestion Lambda — triggered by S3 events on raw/ prefix
resource "aws_lambda_function" "ingestion" {
  count = var.enable_lambda ? 1 : 0

  function_name = "${var.name_prefix}-ingestion"
  package_type  = "Image"
  image_uri     = var.image_uri
  role          = var.lambda_exec_role_arn
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

  tags = { Name = "${var.name_prefix}-ingestion-lambda" }
}

# Allow S3 to invoke the ingestion Lambda
resource "aws_lambda_permission" "s3_invoke_ingestion" {
  count = var.enable_lambda ? 1 : 0

  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingestion[0].function_name
  principal     = "s3.amazonaws.com"
  source_arn    = var.s3_bucket_arn
}

# S3 event notification: raw/ prefix → ingestion Lambda
resource "aws_s3_bucket_notification" "raw_upload" {
  count  = var.enable_lambda ? 1 : 0
  bucket = var.s3_bucket_id

  lambda_function {
    lambda_function_arn = aws_lambda_function.ingestion[0].arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "raw/"
  }

  depends_on = [aws_lambda_permission.s3_invoke_ingestion]
}
