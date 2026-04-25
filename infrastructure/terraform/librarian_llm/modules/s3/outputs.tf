output "bucket_id" {
  description = "S3 bucket name"
  value       = aws_s3_bucket.data_lake.id
}

output "bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.data_lake.arn
}

output "readwrite_policy_json" {
  description = "IAM policy JSON granting read/write on this bucket — pass to iam module"
  value       = data.aws_iam_policy_document.s3_readwrite.json
}
