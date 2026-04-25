aws_region   = "eu-west-1"
environment  = "prod"
project_name = "va-agents"

# Compute — scale up for production
adk_cpu    = 2048
adk_memory = 4096
lg_cpu     = 2048
lg_memory  = 4096

desired_count = 2

gemini_model = "gemini-2.5-flash"

# Secrets — set via TF_VAR_* in CI
# gateway_api_key = "..."
# google_api_key  = "..."

enable_https  = true
# acm_cert_arn  = "arn:aws:acm:eu-west-1:ACCOUNT_ID:certificate/..."
alarm_sns_arn = ""
