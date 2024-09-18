provider "google" {
  credentials = file(var.gcloud_creds)
  project     = var.project_id
  region      = var.region
}

# Cloud Run standalone job
resource "google_cloud_run_v2_job" "default" {
  name     = "hello-world-job"
  location = var.region
  deletion_protection = false

  template {
    template {
      containers {
        image = "gcr.io/${var.project_id}/hello-world:latest"
      }
    }
  }
}

# Cloud Run service
resource "google_cloud_run_service" "default" {
  name     = "streamlit-chatbot-service"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/hello-world:latest" # Update this with your Streamlit container image
        ports {
          container_port = 8080
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# # IAM permission to allow public access to the service
# resource "google_project_iam_member" "all_users" {
#   project = var.project_id
#   role    = "roles/run.invoker"
#   member  = "allUsers"
# }

# Cloud Storage Bucket
resource "google_storage_bucket" "bucket" {
  name     = "${var.project_id}-bucket"
  location = var.region
}

output "cloud_run_job_name" {
  value = google_cloud_run_v2_job.default.name
}