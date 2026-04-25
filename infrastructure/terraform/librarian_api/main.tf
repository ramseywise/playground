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
# Modules
# ---------------------------------------------------------------------------

module "vpc" {
  source = "./modules/vpc"

  name_prefix    = local.name_prefix
  container_port = var.container_port
}

module "ecr" {
  source = "./modules/ecr"

  name_prefix = local.name_prefix
  environment = var.environment
}

module "alb" {
  source = "./modules/alb"

  name_prefix    = local.name_prefix
  vpc_id         = module.vpc.vpc_id
  subnet_ids     = module.vpc.subnet_ids
  alb_sg_id      = module.vpc.alb_sg_id
  container_port = var.container_port
  enable_https   = var.enable_https
  acm_cert_arn   = var.acm_cert_arn
}

module "secrets" {
  source = "./modules/secrets"

  name_prefix             = local.name_prefix
  anthropic_api_key       = var.anthropic_api_key
  checkpoint_postgres_url = var.checkpoint_postgres_url
}

module "cloudwatch" {
  source = "./modules/cloudwatch"

  name_prefix             = local.name_prefix
  aws_region              = var.aws_region
  alb_arn_suffix          = module.alb.alb_arn_suffix
  target_group_arn_suffix = module.alb.target_group_arn_suffix
  ecs_cluster_name        = module.ecs.cluster_name
  ecs_service_name        = module.ecs.service_name
  alarm_sns_arn           = var.alarm_sns_arn
}

module "ecs" {
  source = "./modules/ecs"

  name_prefix           = local.name_prefix
  aws_region            = var.aws_region
  cpu                   = var.cpu
  memory                = var.memory
  desired_count         = var.desired_count
  container_port        = var.container_port
  image_uri             = "${module.ecr.repository_url}:${var.image_tag}"
  subnet_ids            = module.vpc.subnet_ids
  ecs_sg_id             = module.vpc.ecs_sg_id
  target_group_arn      = module.alb.target_group_arn
  alb_listener_arn      = module.alb.http_listener_arn
  anthropic_api_key_arn = module.secrets.anthropic_api_key_arn
  log_group_name        = module.cloudwatch.log_group_name
  s3_bucket_arn         = var.s3_bucket_arn
}

# ---------------------------------------------------------------------------
# moved{} blocks — preserves state addresses when migrating from the flat
# infra/terraform/ layout.  Run `terraform init -migrate-state` after copying
# infra/terraform/terraform.tfstate to this directory (or bootstrapping S3).
#
# Resources that moved to librarian_llm state (s3.tf + lambda.tf) require
# `terraform state mv` — see docs/in-progress/terraform-restructure/plan.md.
# ---------------------------------------------------------------------------

# vpc.tf → module.vpc
moved { from = aws_vpc.main;                      to = module.vpc.aws_vpc.main }
moved { from = aws_internet_gateway.main;         to = module.vpc.aws_internet_gateway.main }
moved { from = aws_subnet.public;                 to = module.vpc.aws_subnet.public }
moved { from = aws_route_table.public;            to = module.vpc.aws_route_table.public }
moved { from = aws_route.internet;                to = module.vpc.aws_route.internet }
moved { from = aws_route_table_association.public; to = module.vpc.aws_route_table_association.public }

# security.tf → module.vpc (co-located with VPC)
moved { from = aws_security_group.alb; to = module.vpc.aws_security_group.alb }
moved { from = aws_security_group.ecs; to = module.vpc.aws_security_group.ecs }

# ecr.tf → module.ecr
moved { from = aws_ecr_repository.api;      to = module.ecr.aws_ecr_repository.api }
moved { from = aws_ecr_lifecycle_policy.api; to = module.ecr.aws_ecr_lifecycle_policy.api }

# alb.tf → module.alb
moved { from = aws_lb.api;              to = module.alb.aws_lb.api }
moved { from = aws_lb_target_group.api; to = module.alb.aws_lb_target_group.api }
moved { from = aws_lb_listener.http;    to = module.alb.aws_lb_listener.http }
moved { from = aws_lb_listener.https;   to = module.alb.aws_lb_listener.https }

# secrets.tf → module.secrets
moved { from = aws_secretsmanager_secret.anthropic_api_key;         to = module.secrets.aws_secretsmanager_secret.anthropic_api_key }
moved { from = aws_secretsmanager_secret_version.anthropic_api_key; to = module.secrets.aws_secretsmanager_secret_version.anthropic_api_key }
moved { from = aws_secretsmanager_secret.checkpoint_postgres_url;         to = module.secrets.aws_secretsmanager_secret.checkpoint_postgres_url }
moved { from = aws_secretsmanager_secret_version.checkpoint_postgres_url; to = module.secrets.aws_secretsmanager_secret_version.checkpoint_postgres_url }

# ecs.tf → module.ecs + module.cloudwatch
moved { from = aws_ecs_cluster.main;                         to = module.ecs.aws_ecs_cluster.main }
moved { from = aws_iam_role.task_execution;                  to = module.ecs.aws_iam_role.task_execution }
moved { from = aws_iam_role_policy_attachment.task_execution; to = module.ecs.aws_iam_role_policy_attachment.task_execution }
moved { from = aws_iam_role.task;                            to = module.ecs.aws_iam_role.task }
moved { from = aws_iam_role_policy.secrets_read;             to = module.ecs.aws_iam_role_policy.secrets_read }
moved { from = aws_iam_role_policy.task_s3;                  to = module.ecs.aws_iam_role_policy.task_s3[0] }
moved { from = aws_ecs_task_definition.api;                  to = module.ecs.aws_ecs_task_definition.api }
moved { from = aws_ecs_service.api;                          to = module.ecs.aws_ecs_service.api }
moved { from = aws_cloudwatch_log_group.api;                 to = module.cloudwatch.aws_cloudwatch_log_group.api }
