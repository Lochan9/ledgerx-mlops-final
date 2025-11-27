# =============================================================================
# ARTIFACT REGISTRY
# =============================================================================

resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = "${var.app_name}-docker"
  description   = "Docker repository for ${var.app_name}"
  format        = "DOCKER"
  labels        = var.labels
}
