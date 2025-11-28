# =============================================================================
# CLOUD STORAGE BUCKETS
# =============================================================================

resource "google_storage_bucket" "invoices" {
  name          = "${var.project_id}-${var.app_name}-invoices"
  location      = var.region
  force_destroy = true
  uniform_bucket_level_access = true
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
  labels = var.labels
}

resource "google_storage_bucket" "models" {
  name          = "${var.project_id}-${var.app_name}-models"
  location      = var.region
  force_destroy = true
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  labels = var.labels
}

resource "google_storage_bucket" "historical_data" {
  name          = "${var.project_id}-${var.app_name}-historical"
  location      = var.region
  force_destroy = true
  uniform_bucket_level_access = true
  labels        = var.labels
}

resource "google_storage_bucket" "reports" {
  name          = "${var.project_id}-${var.app_name}-reports"
  location      = var.region
  force_destroy = true
  uniform_bucket_level_access = true
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
  labels = var.labels
}
