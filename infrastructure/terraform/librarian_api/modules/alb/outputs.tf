output "alb_dns_name" {
  description = "Public DNS name of the ALB"
  value       = aws_lb.api.dns_name
}

output "alb_arn_suffix" {
  description = "ALB ARN suffix for CloudWatch alarms"
  value       = aws_lb.api.arn_suffix
}

output "target_group_arn" {
  description = "Target group ARN for ECS service registration"
  value       = aws_lb_target_group.api.arn
}

output "target_group_arn_suffix" {
  description = "Target group ARN suffix for CloudWatch alarms"
  value       = aws_lb_target_group.api.arn_suffix
}

output "http_listener_arn" {
  description = "HTTP listener ARN — used for ECS service depends_on"
  value       = aws_lb_listener.http.arn
}
