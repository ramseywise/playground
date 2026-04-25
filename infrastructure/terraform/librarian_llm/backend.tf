terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Same S3 bucket as librarian_api, different state key.
  # Bootstrap instructions: see librarian_api/backend.tf
  # State migration: terraform state mv commands required — see plan.md Step 3
  backend "s3" {
    bucket         = "librarian-tfstate"
    key            = "librarian_llm/dev.tfstate"
    region         = "eu-west-1"
    dynamodb_table = "librarian-tfstate-lock"
    encrypt        = true
  }
}
