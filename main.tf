provider "google" {
  credentials = file(var.gcloud_creds)
  project     = var.project_id
  region      = var.region
}

# resource "google_cloud_run_job" "default" {
#   name     = "hello-world-job"
#   location = var.region

#   template {
#     spec {
#       containers {
#         image = "gcr.io/${var.project_id}/hello-world:latest"
#         resources {
#           limits = {
#             memory = "512Mi"
#             cpu    = "1"
#           }
#         }
#         # command = ["python"]
#         # args    = ["HelloWorld.py"]
#       }
#     }
#   }

  # Optionally, configure retry settings and other job parameters here
# }

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

resource "google_project_iam_member" "run_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "user:allAuthenticatedUsers"
}

output "cloud_run_job_name" {
  value = google_cloud_run_v2_job.default.name
}