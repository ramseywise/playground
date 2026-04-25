output "efs_id" {
  description = "EFS file system ID"
  value       = aws_efs_file_system.billy.id
}

output "efs_access_point_id" {
  description = "EFS access point ID for /billy-data"
  value       = aws_efs_access_point.billy.id
}
