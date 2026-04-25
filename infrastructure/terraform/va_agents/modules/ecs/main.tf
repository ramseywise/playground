resource "aws_ecs_cluster" "main" {
  name = "${var.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = { Name = "${var.name_prefix}-cluster" }
}

# ── IAM ──────────────────────────────────────────────────────────────────────

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "task_execution" {
  name               = "${var.name_prefix}-task-exec"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "task_execution" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "secrets_read" {
  statement {
    actions = ["secretsmanager:GetSecretValue"]
    resources = [
      var.gateway_api_key_arn,
      var.google_api_key_arn,
      var.postgres_url_secret_arn,
    ]
  }
}

resource "aws_iam_role_policy" "secrets_read" {
  name   = "${var.name_prefix}-secrets-read"
  role   = aws_iam_role.task_execution.id
  policy = data.aws_iam_policy_document.secrets_read.json
}

resource "aws_iam_role" "task" {
  name               = "${var.name_prefix}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

data "aws_iam_policy_document" "efs_mount" {
  statement {
    actions   = ["elasticfilesystem:ClientMount", "elasticfilesystem:ClientWrite"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "efs_mount" {
  name   = "${var.name_prefix}-efs-mount"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.efs_mount.json
}

# ── Shared billy-mcp sidecar definition ─────────────────────────────────────
#
# Both gateway task definitions embed an identical billy-mcp container.
# Containers in the same Fargate task share a network namespace, so the
# gateway reaches billy at localhost:8765 (MCP SSE) and localhost:8766 (REST).

locals {
  billy_sidecar = {
    name      = "billy-mcp"
    image     = var.billy_mcp_image_uri
    essential = true

    portMappings = [
      { containerPort = 8765, protocol = "tcp" },
      { containerPort = 8766, protocol = "tcp" },
    ]

    environment = [
      { name = "BILLY_DB", value = "/data/billy.db" },
      { name = "MCP_HOST", value = "0.0.0.0" },
      { name = "MCP_PORT", value = "8765" },
      { name = "API_PORT", value = "8766" },
    ]

    mountPoints = [{
      sourceVolume  = "billy-data"
      containerPath = "/data"
      readOnly      = false
    }]

    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8766/docs')\""]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 15
    }

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = var.billy_log_group_name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "billy-mcp"
      }
    }
  }
}

# ── ADK task definition ──────────────────────────────────────────────────────

resource "aws_ecs_task_definition" "adk" {
  family                   = "${var.name_prefix}-adk"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.adk_cpu
  memory                   = var.adk_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  volume {
    name = "billy-data"
    efs_volume_configuration {
      file_system_id          = var.efs_id
      transit_encryption      = "ENABLED"
      authorization_config {
        access_point_id = var.efs_access_point_id
        iam             = "ENABLED"
      }
    }
  }

  container_definitions = jsonencode([
    {
      name      = "va-gateway-adk"
      image     = var.adk_image_uri
      essential = true

      portMappings = [{ containerPort = 8000, protocol = "tcp" }]

      environment = [
        { name = "GATEWAY_HOST", value = "0.0.0.0" },
        { name = "GATEWAY_PORT", value = "8000" },
        { name = "GEMINI_MODEL", value = var.gemini_model },
        { name = "BILLY_MCP_URL", value = "http://localhost:8765/sse" },
        { name = "BILLY_API_URL", value = "http://localhost:8766" },
      ]

      secrets = [
        { name = "GATEWAY_API_KEY", valueFrom = var.gateway_api_key_arn },
        { name = "GOOGLE_API_KEY", valueFrom = var.google_api_key_arn },
      ]

      dependsOn = [{ containerName = "billy-mcp", condition = "HEALTHY" }]

      healthCheck = {
        command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\""]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = var.adk_log_group_name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "gateway-adk"
        }
      }
    },
    local.billy_sidecar,
  ])
}

# ── LG task definition ───────────────────────────────────────────────────────

resource "aws_ecs_task_definition" "lg" {
  family                   = "${var.name_prefix}-lg"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.lg_cpu
  memory                   = var.lg_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  volume {
    name = "billy-data"
    efs_volume_configuration {
      file_system_id          = var.efs_id
      transit_encryption      = "ENABLED"
      authorization_config {
        access_point_id = var.efs_access_point_id
        iam             = "ENABLED"
      }
    }
  }

  container_definitions = jsonencode([
    {
      name      = "va-gateway-lg"
      image     = var.lg_image_uri
      essential = true

      portMappings = [{ containerPort = 8001, protocol = "tcp" }]

      environment = [
        { name = "GATEWAY_HOST", value = "0.0.0.0" },
        { name = "GATEWAY_PORT", value = "8001" },
        { name = "GEMINI_MODEL", value = var.gemini_model },
        { name = "BILLY_MCP_URL", value = "http://localhost:8765/sse" },
        { name = "BILLY_API_URL", value = "http://localhost:8766" },
        { name = "LANGGRAPH_CHECKPOINTER", value = "postgres" },
      ]

      secrets = [
        { name = "GATEWAY_API_KEY", valueFrom = var.gateway_api_key_arn },
        { name = "GOOGLE_API_KEY", valueFrom = var.google_api_key_arn },
        { name = "POSTGRES_URL", valueFrom = var.postgres_url_secret_arn },
      ]

      dependsOn = [{ containerName = "billy-mcp", condition = "HEALTHY" }]

      healthCheck = {
        command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8001/health')\""]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = var.lg_log_group_name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "gateway-lg"
        }
      }
    },
    local.billy_sidecar,
  ])
}

# ── ECS services ─────────────────────────────────────────────────────────────

resource "aws_ecs_service" "adk" {
  name            = "${var.name_prefix}-adk"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.adk.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [var.ecs_sg_id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = var.adk_target_group_arn
    container_name   = "va-gateway-adk"
    container_port   = 8000
  }

  depends_on = [var.adk_listener_arn]
}

resource "aws_ecs_service" "lg" {
  name            = "${var.name_prefix}-lg"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.lg.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [var.ecs_sg_id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = var.lg_target_group_arn
    container_name   = "va-gateway-lg"
    container_port   = 8001
  }

  depends_on = [var.lg_listener_arn]
}
