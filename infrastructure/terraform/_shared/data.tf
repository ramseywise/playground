# ---------------------------------------------------------------------------
# Cross-stack data source: read librarian_api outputs from S3 remote state.
# Sourced by librarian_llm to get ECR repository URL for Lambda image URI.
#
# Deployment order: librarian_api must be applied (and state pushed to S3)
# before librarian_llm can read its outputs here.
# ---------------------------------------------------------------------------

data "terraform_remote_state" "librarian_api" {
  backend = "s3"
  config = {
    bucket = "librarian-tfstate"
    key    = "librarian_api/${var.environment}.tfstate"
    region = "eu-west-1"
  }
}
