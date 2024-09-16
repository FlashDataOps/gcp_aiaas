provider "google" {
  credentials = file(var.gcloud_creds)
  project     = var.project_id
  region      = var.region
}

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

output "cloud_run_job_name" {
  value = google_cloud_run_v2_job.default.name
}