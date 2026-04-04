variable "app_name" {
  description = "Application name"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "bucket_suffix" {
  description = "Suffix for the bucket name (e.g., 'data', 'logs', 'backups')"
  type        = string
  default     = "bucket"
}

variable "kms_key_id" {
  description = "KMS key ID for bucket encryption (empty string uses AES256)"
  type        = string
  default     = ""
}

variable "enable_lifecycle" {
  description = "Enable S3 lifecycle rules for cost optimization"
  type        = bool
  default     = true
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default     = {}
}
