# =============================================================================
# TERRAFORM VARIABLES
# =============================================================================

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "app_name" {
  description = "Application name"
  type        = string
  default     = "ledgerx"
}

variable "environment" {
  description = "Environment"
  type        = string
  default     = "prod"
}

variable "docker_image" {
  description = "Docker image name"
  type        = string
  default     = "ledgerx-api"
}

variable "container_port" {
  description = "Container port"
  type        = number
  default     = 8000
}

variable "cloud_run_cpu" {
  description = "CPU allocation"
  type        = string
  default     = "1"
}

variable "cloud_run_memory" {
  description = "Memory allocation"
  type        = string
  default     = "512Mi"
}

variable "cloud_run_min_instances" {
  description = "Min instances"
  type        = number
  default     = 0
}

variable "cloud_run_max_instances" {
  description = "Max instances"
  type        = number
  default     = 3
}

variable "cloud_run_timeout" {
  description = "Request timeout"
  type        = number
  default     = 60
}

variable "enable_public_access" {
  description = "Allow public access"
  type        = bool
  default     = true
}

variable "labels" {
  description = "Resource labels"
  type        = map(string)
  default = {
    app         = "ledgerx"
    managed_by  = "terraform"
    environment = "prod"
  }
}
