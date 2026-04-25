# ---------------------------------------------------------------------------
# ECS Cluster + Fargate Service
# ---------------------------------------------------------------------------

resource "aws_ecs_cluster" "main" {
  name = "${var.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = { Name = "${var.name_prefix}-cluster" }
}

# ---------------------------------------------------------------------------
# IAM — task execution role (pull ECR images, read secrets)
# ---------------------------------------------------------------------------

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

# Allow reading secrets
data "aws_iam_policy_document" "secrets_read" {
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [var.anthropic_api_key_arn]
  }
}

resource "aws_iam_role_policy" "secrets_read" {
  name   = "${var.name_prefix}-secrets-read"
  role   = aws_iam_role.task_execution.id
  policy = data.aws_iam_policy_document.secrets_read.json
}

# ---------------------------------------------------------------------------
# IAM — task role (for the running container itself)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "task" {
  name               = "${var.name_prefix}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

# S3 read/write for data lake — only attached when s3_bucket_arn is provided.
# Set var.s3_bucket_arn after librarian_llm stack is deployed.
data "aws_iam_policy_document" "task_s3" {
  count = var.s3_bucket_arn != "" ? 1 : 0

  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
      "s3:DeleteObject",
    ]
    resources = [
      var.s3_bucket_arn,
      "${var.s3_bucket_arn}/*",
    ]
  }
}

resource "aws_iam_role_policy" "task_s3" {
  count = var.s3_bucket_arn != "" ? 1 : 0

  name   = "${var.name_prefix}-task-s3"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_s3[0].json
}

# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

resource "aws_ecs_task_definition" "api" {
  family                   = "${var.name_prefix}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name  = "api"
    image = var.image_uri

    portMappings = [{
      containerPort = var.container_port
      protocol      = "tcp"
    }]

    environment = [
      { name = "RETRIEVAL_STRATEGY", value = "chroma" },
      { name = "RERANKER_STRATEGY", value = "cross_encoder" },
      { name = "CONFIDENCE_THRESHOLD", value = "0.4" },
      { name = "API_HOST", value = "0.0.0.0" },
      { name = "API_PORT", value = tostring(var.container_port) },
    ]

    secrets = [{
      name      = "ANTHROPIC_API_KEY"
      valueFrom = var.anthropic_api_key_arn
    }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = var.log_group_name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "api"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import httpx; httpx.get('http://localhost:${var.container_port}/health').raise_for_status()\""]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60 # multilingual-e5-large loads ~45s on cold start
    }
  }])
}

# ---------------------------------------------------------------------------
# ECS Service
# ---------------------------------------------------------------------------

resource "aws_ecs_service" "api" {
  name            = "${var.name_prefix}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [var.ecs_sg_id]
    assign_public_ip = true # public subnet, no NAT
  }

  load_balancer {
    target_group_arn = var.target_group_arn
    container_name   = "api"
    container_port   = var.container_port
  }

  # Implicit ordering: target_group_arn comes from module.alb, so Terraform
  # applies the ALB (and its listener) before this service. No explicit
  # depends_on needed.
}
