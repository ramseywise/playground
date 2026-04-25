terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "librarian-tfstate"
    key            = "va_agents/dev.tfstate"
    region         = "eu-west-1"
    dynamodb_table = "librarian-tfstate-lock"
    encrypt        = true
  }
}
