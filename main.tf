provider "google" {
  credentials = file(var.gcloud_creds)
  project = var.project_id
  region  = var.region
}

resource "google_cloud_run_service" "default" {
  name     = "hello-world"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/hello-world:latest"
        resources {
          limits = {
            memory = "512Mi"
            cpu    = "1"
          }
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_project_iam_member" "run_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "allUsers"
}

output "cloud_run_url" {
  value = google_cloud_run_service.default.status[0].url
}