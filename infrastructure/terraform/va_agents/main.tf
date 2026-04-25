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

module "vpc" {
  source = "./modules/vpc"

  name_prefix = local.name_prefix
}

module "ecr" {
  source = "./modules/ecr"

  name_prefix = local.name_prefix
  environment = var.environment
}

module "alb" {
  source = "./modules/alb"

  name_prefix  = local.name_prefix
  vpc_id       = module.vpc.vpc_id
  subnet_ids   = module.vpc.subnet_ids
  alb_sg_id    = module.vpc.alb_sg_id
  enable_https = var.enable_https
  acm_cert_arn = var.acm_cert_arn
}

module "secrets" {
  source = "./modules/secrets"

  name_prefix     = local.name_prefix
  gateway_api_key = var.gateway_api_key
  google_api_key  = var.google_api_key
}

module "rds" {
  source = "./modules/rds"

  name_prefix = local.name_prefix
  vpc_id      = module.vpc.vpc_id
  subnet_ids  = module.vpc.subnet_ids
  ecs_sg_id   = module.vpc.ecs_sg_id
  db_password = var.db_password
}

module "efs" {
  source = "./modules/efs"

  name_prefix = local.name_prefix
  vpc_id      = module.vpc.vpc_id
  subnet_ids  = module.vpc.subnet_ids
  ecs_sg_id   = module.vpc.ecs_sg_id
}

module "cloudwatch" {
  source = "./modules/cloudwatch"

  name_prefix                  = local.name_prefix
  aws_region                   = var.aws_region
  alb_arn_suffix               = module.alb.alb_arn_suffix
  adk_target_group_arn_suffix  = module.alb.adk_target_group_arn_suffix
  lg_target_group_arn_suffix   = module.alb.lg_target_group_arn_suffix
  ecs_cluster_name             = module.ecs.cluster_name
  ecs_adk_service_name         = module.ecs.adk_service_name
  ecs_lg_service_name          = module.ecs.lg_service_name
  alarm_sns_arn                = var.alarm_sns_arn
}

module "ecs" {
  source = "./modules/ecs"

  name_prefix            = local.name_prefix
  aws_region             = var.aws_region
  adk_cpu                = var.adk_cpu
  adk_memory             = var.adk_memory
  lg_cpu                 = var.lg_cpu
  lg_memory              = var.lg_memory
  desired_count          = var.desired_count
  adk_image_uri          = "${module.ecr.gateway_adk_url}:${var.adk_image_tag}"
  lg_image_uri           = "${module.ecr.gateway_lg_url}:${var.lg_image_tag}"
  billy_mcp_image_uri    = "${module.ecr.billy_mcp_url}:${var.billy_mcp_image_tag}"
  subnet_ids             = module.vpc.subnet_ids
  ecs_sg_id              = module.vpc.ecs_sg_id
  adk_target_group_arn   = module.alb.adk_target_group_arn
  lg_target_group_arn    = module.alb.lg_target_group_arn
  adk_listener_arn       = module.alb.adk_listener_arn
  lg_listener_arn        = module.alb.lg_listener_arn
  gateway_api_key_arn    = module.secrets.gateway_api_key_arn
  google_api_key_arn     = module.secrets.google_api_key_arn
  postgres_url_secret_arn = module.rds.postgres_url_secret_arn
  efs_id                  = module.efs.efs_id
  efs_access_point_id     = module.efs.efs_access_point_id
  gemini_model           = var.gemini_model
  adk_log_group_name     = module.cloudwatch.adk_log_group_name
  lg_log_group_name      = module.cloudwatch.lg_log_group_name
  billy_log_group_name   = module.cloudwatch.billy_log_group_name
}
