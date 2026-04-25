variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
}

variable "gateway_api_key" {
  description = "API key for gateway X-API-Key header auth"
  type        = string
  sensitive   = true
}

variable "google_api_key" {
  description = "Google / Gemini API key"
  type        = string
  sensitive   = true
}
