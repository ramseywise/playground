resource "aws_cloudwatch_log_group" "adk" {
  name              = "/ecs/${var.name_prefix}-adk"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "lg" {
  name              = "/ecs/${var.name_prefix}-lg"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "billy" {
  name              = "/ecs/${var.name_prefix}-billy-mcp"
  retention_in_days = 14
}

locals {
  alarm_actions = var.alarm_sns_arn != "" ? [var.alarm_sns_arn] : []
}

# ── ADK service alarms ───────────────────────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "adk_5xx_rate" {
  alarm_name          = "${var.name_prefix}-adk-5xx-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  threshold           = 5
  alarm_description   = "ADK gateway ALB 5xx error rate exceeds 5%"
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
        TargetGroup  = var.adk_target_group_arn_suffix
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
        TargetGroup  = var.adk_target_group_arn_suffix
      }
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "adk_task_count" {
  alarm_name          = "${var.name_prefix}-adk-running-tasks"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "RunningTaskCount"
  namespace           = "ECS/ContainerInsights"
  period              = 60
  statistic           = "Average"
  threshold           = 1
  alarm_description   = "ADK gateway ECS running task count is 0"
  alarm_actions       = local.alarm_actions
  treat_missing_data  = "breaching"

  dimensions = {
    ClusterName = var.ecs_cluster_name
    ServiceName = var.ecs_adk_service_name
  }
}

# ── LG service alarms ────────────────────────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "lg_5xx_rate" {
  alarm_name          = "${var.name_prefix}-lg-5xx-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  threshold           = 5
  alarm_description   = "LG gateway ALB 5xx error rate exceeds 5%"
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
        TargetGroup  = var.lg_target_group_arn_suffix
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
        TargetGroup  = var.lg_target_group_arn_suffix
      }
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "lg_task_count" {
  alarm_name          = "${var.name_prefix}-lg-running-tasks"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "RunningTaskCount"
  namespace           = "ECS/ContainerInsights"
  period              = 60
  statistic           = "Average"
  threshold           = 1
  alarm_description   = "LG gateway ECS running task count is 0"
  alarm_actions       = local.alarm_actions
  treat_missing_data  = "breaching"

  dimensions = {
    ClusterName = var.ecs_cluster_name
    ServiceName = var.ecs_lg_service_name
  }
}
