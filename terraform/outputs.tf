# =============================================================================
# OUTPUTS
# =============================================================================

output "api_endpoint" {
  description = "Your LedgerX API URL"
  value       = google_cloud_run_service.ledgerx_api.status[0].url
}

output "api_docs" {
  description = "API documentation URL"
  value       = "${google_cloud_run_service.ledgerx_api.status[0].url}/docs"
}

output "docker_repository_url" {
  description = "Docker repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}"
}

output "storage_buckets" {
  description = "Storage buckets"
  value = {
    invoices   = google_storage_bucket.invoices.name
    models     = google_storage_bucket.models.name
    historical = google_storage_bucket.historical_data.name
    reports    = google_storage_bucket.reports.name
  }
}

output "next_steps" {
  description = "What to do next"
  value = <<-EOT
  
  ✅ DEPLOYMENT COMPLETE!
  
  1. Build Docker: docker build -t ${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}/${var.docker_image}:latest .
  2. Push: docker push ${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}/${var.docker_image}:latest
  3. Test: curl ${google_cloud_run_service.ledgerx_api.status[0].url}/health
  
  API URL: ${google_cloud_run_service.ledgerx_api.status[0].url}
  
  EOT
}
