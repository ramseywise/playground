output "alb_dns_name" {
  description = "Public DNS name of the ALB"
  value       = aws_lb.va.dns_name
}

output "alb_arn_suffix" {
  description = "ALB ARN suffix for CloudWatch alarms"
  value       = aws_lb.va.arn_suffix
}

output "adk_target_group_arn" {
  description = "ADK target group ARN"
  value       = aws_lb_target_group.adk.arn
}

output "adk_target_group_arn_suffix" {
  description = "ADK target group ARN suffix for CloudWatch alarms"
  value       = aws_lb_target_group.adk.arn_suffix
}

output "lg_target_group_arn" {
  description = "LG target group ARN"
  value       = aws_lb_target_group.lg.arn
}

output "lg_target_group_arn_suffix" {
  description = "LG target group ARN suffix for CloudWatch alarms"
  value       = aws_lb_target_group.lg.arn_suffix
}

output "adk_listener_arn" {
  description = "ADK HTTP listener ARN"
  value       = aws_lb_listener.adk_http.arn
}

output "lg_listener_arn" {
  description = "LG listener ARN (port 8001)"
  value       = aws_lb_listener.lg.arn
}
