# ---------------------------------------------------------------------------
# CloudWatch — log group (extracted from ecs.tf) + operational alarms
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.name_prefix}"
  retention_in_days = 14
}

locals {
  alarm_actions = var.alarm_sns_arn != "" ? [var.alarm_sns_arn] : []
}

# ---------------------------------------------------------------------------
# Alarms (Step 5 additions — additive only, no resource address changes)
# ---------------------------------------------------------------------------

# ALB 5xx error rate > 5% over 5 minutes
resource "aws_cloudwatch_metric_alarm" "alb_5xx_rate" {
  alarm_name          = "${var.name_prefix}-alb-5xx-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  threshold           = 5
  alarm_description   = "ALB 5xx error rate exceeds 5% — possible app crash or upstream error"
  alarm_actions       = local.alarm_actions
  treat_missing_data  = "notBreaching"

  metric_query {
    id          = "error_rate"
    expression  = "100 * errors / MAX([errors, requests])"
    label       = "5xx Error Rate (%)"
    return_data = true
  }

  metric_query {
    id = "errors"
    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "HTTPCode_Target_5XX_Count"
      period      = 300
      stat        = "Sum"
      dimensions = {
        LoadBalancer = var.alb_arn_suffix
        TargetGroup  = var.target_group_arn_suffix
      }
    }
  }

  metric_query {
    id = "requests"
    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "RequestCount"
      period      = 300
      stat        = "Sum"
      dimensions = {
        LoadBalancer = var.alb_arn_suffix
        TargetGroup  = var.target_group_arn_suffix
      }
    }
  }
}

# ALB target response time p99 > 2 seconds
resource "aws_cloudwatch_metric_alarm" "alb_latency_p99" {
  alarm_name          = "${var.name_prefix}-alb-latency-p99"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = 300
  extended_statistic  = "p99"
  threshold           = 2
  alarm_description   = "ALB p99 latency exceeds 2s — check ECS task performance"
  alarm_actions       = local.alarm_actions
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = var.alb_arn_suffix
    TargetGroup  = var.target_group_arn_suffix
  }
}

# ECS service task count drops to 0
resource "aws_cloudwatch_metric_alarm" "ecs_task_count" {
  alarm_name          = "${var.name_prefix}-ecs-running-tasks"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "RunningTaskCount"
  namespace           = "ECS/ContainerInsights"
  period              = 60
  statistic           = "Average"
  threshold           = 1
  alarm_description   = "ECS running task count is 0 — service may be down"
  alarm_actions       = local.alarm_actions
  treat_missing_data  = "breaching"

  dimensions = {
    ClusterName = var.ecs_cluster_name
    ServiceName = var.ecs_service_name
  }
}
