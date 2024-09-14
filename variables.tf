variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The region where Cloud Run will be deployed"
  type        = string
  default     = "us-central1"
}