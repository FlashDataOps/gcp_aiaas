variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The region where Cloud Run will be deployed"
  type        = string
  default     = "us-central1"
}

variable "gcloud_creds" {
  description = "The path to the Google Cloud credentials file or the credentials JSON content"
  type        = string
}

variable "image_tag" {
  description = "The docker image tag"
  type        = string
}