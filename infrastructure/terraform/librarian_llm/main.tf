provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# ---------------------------------------------------------------------------
# Cross-stack: read librarian_api outputs to get ECR repository URL
# Deployment order: librarian_api must be applied first.
# ---------------------------------------------------------------------------

module "shared" {
  source      = "../_shared"
  environment = var.environment
}

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

module "s3" {
  source = "./modules/s3"

  name_prefix = local.name_prefix
  environment = var.environment
}

module "iam" {
  source = "./modules/iam"

  name_prefix              = local.name_prefix
  enable_lambda            = var.enable_lambda
  s3_readwrite_policy_json = module.s3.readwrite_policy_json
  anthropic_api_key_arn    = var.anthropic_api_key_arn
}

module "lambda" {
  source = "./modules/lambda"

  name_prefix          = local.name_prefix
  enable_lambda        = var.enable_lambda
  image_uri            = "${module.shared.ecr_repository_url}:${var.image_tag}"
  lambda_memory        = var.lambda_memory
  lambda_timeout       = var.lambda_timeout
  lambda_auth_type     = var.lambda_auth_type
  lambda_exec_role_arn = module.iam.lambda_exec_role_arn
  s3_bucket_arn        = module.s3.bucket_arn
  s3_bucket_id         = module.s3.bucket_id
}

# ---------------------------------------------------------------------------
# moved{} blocks — resources migrating from librarian_api flat state.
#
# These blocks alone are NOT sufficient — resources must first be moved
# between state files using `terraform state mv`.  Run these commands
# against the ORIGINAL infra/terraform/terraform.tfstate:
#
#   terraform -chdir=infra/terraform state mv \
#     -state-out=../../infrastructure/infrastructure_as_code/librarian_llm/terraform.tfstate \
#     aws_s3_bucket.data_lake \
#     module.s3.aws_s3_bucket.data_lake
#
# (repeat for each resource below, adjusting source and target addresses)
#
# Full list of resources to migrate:
#   aws_s3_bucket.data_lake                             → module.s3.aws_s3_bucket.data_lake
#   aws_s3_bucket_versioning.data_lake                  → module.s3.aws_s3_bucket_versioning.data_lake
#   aws_s3_bucket_server_side_encryption_configuration.data_lake → module.s3.aws_s3_bucket_server_side_encryption_configuration.data_lake
#   aws_s3_bucket_public_access_block.data_lake         → module.s3.aws_s3_bucket_public_access_block.data_lake
#   aws_s3_bucket_notification.raw_upload               → module.lambda.aws_s3_bucket_notification.raw_upload[0]
#   aws_iam_role.lambda_exec                            → module.iam.aws_iam_role.lambda_exec[0]
#   aws_iam_role_policy_attachment.lambda_basic         → module.iam.aws_iam_role_policy_attachment.lambda_basic[0]
#   aws_iam_role_policy.lambda_secrets                  → module.iam.aws_iam_role_policy.lambda_secrets[0]
#   aws_iam_role_policy.lambda_s3                       → module.iam.aws_iam_role_policy.lambda_s3[0]
#   aws_lambda_function.api                             → module.lambda.aws_lambda_function.api[0]
#   aws_lambda_function_url.api                         → module.lambda.aws_lambda_function_url.api[0]
#   aws_lambda_function.ingestion                       → module.lambda.aws_lambda_function.ingestion[0]
#   aws_lambda_permission.s3_invoke_ingestion           → module.lambda.aws_lambda_permission.s3_invoke_ingestion[0]
# ---------------------------------------------------------------------------
