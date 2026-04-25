terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # ---------------------------------------------------------------------------
  # S3 remote backend — Step 4 migration
  # ---------------------------------------------------------------------------
  # Bootstrap one-liner (run once before `terraform init -migrate-state`):
  #
  #   aws s3api create-bucket \
  #     --bucket librarian-tfstate \
  #     --region eu-west-1 \
  #     --create-bucket-configuration LocationConstraint=eu-west-1
  #
  #   aws s3api put-bucket-versioning \
  #     --bucket librarian-tfstate \
  #     --versioning-configuration Status=Enabled
  #
  #   aws dynamodb create-table \
  #     --table-name librarian-tfstate-lock \
  #     --attribute-definitions AttributeName=LockID,AttributeType=S \
  #     --key-schema AttributeName=LockID,KeyType=HASH \
  #     --billing-mode PAY_PER_REQUEST \
  #     --region eu-west-1
  #
  # Then copy infra/terraform/terraform.tfstate to this directory and run:
  #   terraform init -migrate-state
  # ---------------------------------------------------------------------------
  backend "s3" {
    bucket         = "librarian-tfstate"
    key            = "librarian_api/dev.tfstate"
    region         = "eu-west-1"
    dynamodb_table = "librarian-tfstate-lock"
    encrypt        = true
  }
}
