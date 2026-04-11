# ---------------------------------------------------------------------------
# S3 Data Lake — raw docs + processed artifacts
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "data_lake" {
  bucket        = "${local.name_prefix}-data-lake"
  force_destroy = var.environment == "dev"

  tags = { Name = "${local.name_prefix}-data-lake" }
}

resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Shared IAM policy — used by both ECS task role and Lambda role
data "aws_iam_policy_document" "s3_readwrite" {
  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
      "s3:DeleteObject",
    ]
    resources = [
      aws_s3_bucket.data_lake.arn,
      "${aws_s3_bucket.data_lake.arn}/*",
    ]
  }
}

# S3 event notification: raw/ prefix → ingestion Lambda
resource "aws_s3_bucket_notification" "raw_upload" {
  count  = var.enable_lambda ? 1 : 0
  bucket = aws_s3_bucket.data_lake.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.ingestion[0].arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "raw/"
  }

  depends_on = [aws_lambda_permission.s3_invoke_ingestion]
}
