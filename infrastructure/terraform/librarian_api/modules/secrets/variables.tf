variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
}

variable "anthropic_api_key" {
  description = "Anthropic API key — stored in Secrets Manager"
  type        = string
  sensitive   = true
}

variable "checkpoint_postgres_url" {
  description = "Postgres connection URL for LangGraph checkpointer — stored in Secrets Manager"
  type        = string
  sensitive   = true
  default     = ""
}
