# =============================================================================
# CLOUD RUN SERVICE
# =============================================================================

resource "google_cloud_run_service" "ledgerx_api" {
  name     = "${var.app_name}-api"
  location = var.region
  
  template {
    metadata {
      labels = var.labels
      annotations = {
        "autoscaling.knative.dev/minScale"   = tostring(var.cloud_run_min_instances)
        "autoscaling.knative.dev/maxScale"   = tostring(var.cloud_run_max_instances)
        "run.googleapis.com/service-account" = google_service_account.cloud_run_sa.email
        "run.googleapis.com/cpu-throttling"  = "true"
      }
    }
    
    spec {
      containers {
        image = local.docker_image_url
        
        resources {
          limits = {
            cpu    = var.cloud_run_cpu
            memory = var.cloud_run_memory
          }
        }
        
        ports {
          container_port = var.container_port
        }
        
        env {
          name  = "ENVIRONMENT"
          value = var.environment
        }
        
        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }
        
        env {
          name  = "INVOICES_BUCKET"
          value = google_storage_bucket.invoices.name
        }
        
        env {
          name  = "MODELS_BUCKET"
          value = google_storage_bucket.models.name
        }
        
        env {
          name  = "HISTORICAL_BUCKET"
          value = google_storage_bucket.historical_data.name
        }
        
        startup_probe {
          http_get {
            path = "/health"
            port = var.container_port
          }
          initial_delay_seconds = 10
          timeout_seconds       = 3
          period_seconds        = 10
          failure_threshold     = 3
        }
        
        liveness_probe {
          http_get {
            path = "/health"
            port = var.container_port
          }
          initial_delay_seconds = 30
          timeout_seconds       = 3
          period_seconds        = 30
        }
      }
      
      timeout_seconds      = var.cloud_run_timeout
      service_account_name = google_service_account.cloud_run_sa.email
    }
  }
  
  traffic {
    percent         = 100
    latest_revision = true
  }
  
  depends_on = [
    google_artifact_registry_repository.docker_repo,
    google_service_account.cloud_run_sa
  ]
}

resource "google_cloud_run_service_iam_member" "public_access" {
  count    = var.enable_public_access ? 1 : 0
  service  = google_cloud_run_service.ledgerx_api.name
  location = google_cloud_run_service.ledgerx_api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
