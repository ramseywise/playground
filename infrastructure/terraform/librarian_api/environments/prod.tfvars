# librarian_api — prod environment
# All sensitive values via CI environment variables:
#   TF_VAR_anthropic_api_key, TF_VAR_checkpoint_postgres_url

aws_region   = "eu-west-1"
environment  = "prod"
project_name = "librarian"

# ECS sizing — scale up for prod
cpu           = 1024
memory        = 4096
desired_count = 2
container_port = 8000

# image_tag set at deploy time by CI pipeline
# image_tag = "<git-sha>"

# HTTPS — set acm_cert_arn once cert is provisioned
enable_https = false
acm_cert_arn = ""

# S3 bucket ARN from librarian_llm prod stack
s3_bucket_arn = "arn:aws:s3:::librarian-prod-data-lake"

# SNS topic for oncall paging
alarm_sns_arn = ""
