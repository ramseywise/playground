resource "aws_lb" "va" {
  name               = "${var.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [var.alb_sg_id]
  subnets            = var.subnet_ids

  # SSE streams require a long idle timeout; 300s matches the gateway's keepalive
  idle_timeout = 300

  tags = { Name = "${var.name_prefix}-alb" }
}

# ── ADK target group (port 8000) ────────────────────────────────────────────

resource "aws_lb_target_group" "adk" {
  name        = "${var.name_prefix}-adk-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path                = "/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200"
  }

  tags = { Name = "${var.name_prefix}-adk-tg" }
}

# ── LG target group (port 8001) ─────────────────────────────────────────────

resource "aws_lb_target_group" "lg" {
  name        = "${var.name_prefix}-lg-tg"
  port        = 8001
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path                = "/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200"
  }

  tags = { Name = "${var.name_prefix}-lg-tg" }
}

# ── Listeners ────────────────────────────────────────────────────────────────

resource "aws_lb_listener" "adk_http" {
  load_balancer_arn = aws_lb.va.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = var.enable_https ? "redirect" : "forward"

    dynamic "redirect" {
      for_each = var.enable_https ? [1] : []
      content {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }

    dynamic "forward" {
      for_each = var.enable_https ? [] : [1]
      content {
        target_group {
          arn = aws_lb_target_group.adk.arn
        }
      }
    }
  }
}

resource "aws_lb_listener" "adk_https" {
  count = var.enable_https ? 1 : 0

  load_balancer_arn = aws_lb.va.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.acm_cert_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.adk.arn
  }
}

# LG gateway on its own port so no path rewriting is needed
resource "aws_lb_listener" "lg" {
  load_balancer_arn = aws_lb.va.arn
  port              = 8001
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.lg.arn
  }
}
