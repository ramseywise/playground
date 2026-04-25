# librarian_llm — prod environment

aws_region   = "eu-west-1"
environment  = "prod"
project_name = "librarian"

# Lambda disabled until production validation is complete
enable_lambda    = false
lambda_memory    = 1024
lambda_timeout   = 60
lambda_auth_type = "AWS_IAM"

# image_tag set at deploy time by CI pipeline
# image_tag = "<git-sha>"

# Anthropic API key ARN from librarian_api prod stack outputs
anthropic_api_key_arn = ""
