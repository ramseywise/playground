# librarian_api — dev environment
# Pass secrets via env vars or CI secret manager, never commit values here:
#   export TF_VAR_anthropic_api_key=sk-ant-...
#   export TF_VAR_checkpoint_postgres_url=postgresql://...

aws_region   = "eu-west-1"
environment  = "dev"
project_name = "librarian"

# ECS sizing — minimal for dev
cpu           = 512
memory        = 4096
desired_count = 1
container_port = 8000

# Image tag — override at apply time with -var="image_tag=$(git rev-parse --short HEAD)"
image_tag = "latest"

# HTTPS disabled in dev (no ACM cert)
enable_https = false

# S3 bucket ARN from librarian_llm stack — set after llm stack is deployed:
#   s3_bucket_arn = "arn:aws:s3:::librarian-dev-data-lake"
s3_bucket_arn = ""

# SNS alarm topic — leave empty to fire silently in dev
alarm_sns_arn = ""
