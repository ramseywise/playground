locals {
  repos = {
    gateway_adk = "${var.name_prefix}-gateway-adk"
    gateway_lg  = "${var.name_prefix}-gateway-lg"
    billy_mcp   = "${var.name_prefix}-billy-mcp"
  }

  lifecycle_policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_ecr_repository" "gateway_adk" {
  name                 = local.repos.gateway_adk
  image_tag_mutability = "MUTABLE"
  force_delete         = var.environment == "dev"

  image_scanning_configuration { scan_on_push = true }

  tags = { Name = local.repos.gateway_adk }
}

resource "aws_ecr_lifecycle_policy" "gateway_adk" {
  repository = aws_ecr_repository.gateway_adk.name
  policy     = local.lifecycle_policy
}

resource "aws_ecr_repository" "gateway_lg" {
  name                 = local.repos.gateway_lg
  image_tag_mutability = "MUTABLE"
  force_delete         = var.environment == "dev"

  image_scanning_configuration { scan_on_push = true }

  tags = { Name = local.repos.gateway_lg }
}

resource "aws_ecr_lifecycle_policy" "gateway_lg" {
  repository = aws_ecr_repository.gateway_lg.name
  policy     = local.lifecycle_policy
}

resource "aws_ecr_repository" "billy_mcp" {
  name                 = local.repos.billy_mcp
  image_tag_mutability = "MUTABLE"
  force_delete         = var.environment == "dev"

  image_scanning_configuration { scan_on_push = true }

  tags = { Name = local.repos.billy_mcp }
}

resource "aws_ecr_lifecycle_policy" "billy_mcp" {
  repository = aws_ecr_repository.billy_mcp.name
  policy     = local.lifecycle_policy
}
