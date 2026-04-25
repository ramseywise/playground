# librarian_llm — dev environment

aws_region   = "eu-west-1"
environment  = "dev"
project_name = "librarian"

# Lambda disabled by default — enable when testing serverless path
enable_lambda    = false
lambda_memory    = 1024
lambda_timeout   = 60
lambda_auth_type = "NONE"

image_tag = "latest"

# Anthropic API key ARN from librarian_api stack outputs — set after api stack deployed:
#   anthropic_api_key_arn = "arn:aws:secretsmanager:eu-west-1:...:secret:librarian-dev/anthropic-api-key-..."
anthropic_api_key_arn = ""
