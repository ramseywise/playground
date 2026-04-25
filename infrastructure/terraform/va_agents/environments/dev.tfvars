aws_region   = "eu-west-1"
environment  = "dev"
project_name = "va-agents"

# Compute — keep dev light
adk_cpu    = 1024
adk_memory = 2048
lg_cpu     = 1024
lg_memory  = 2048

desired_count = 1

gemini_model = "gemini-2.5-flash-lite"

# Secrets — values set via TF_VAR_* env vars or CI secret injection, never committed
# gateway_api_key = "..."
# google_api_key  = "..."

enable_https = false
alarm_sns_arn = ""
