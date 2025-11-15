# Production Environment Configuration
terraform {
  backend "gcs" {
    bucket = "sie-x-terraform-state"
    prefix = "production"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

module "sie_x" {
  source = "../../modules/sie-x"

  project_id  = var.project_id
  region      = var.region
  environment = "prod"

  sie_x_config = {
    version          = "v3.0.0"
    replicas         = 5
    gpu_enabled      = true
    gpu_type         = "nvidia-tesla-v100"
    max_gpu_per_node = 4
    memory_gb        = 32
    cpu_cores        = 16
    models           = [
      "sentence-transformers/all-mpnet-base-v2",
      "sentence-transformers/LaBSE",
      "xlm-roberta-large",
      "google/mt5-xl"
    ]
    cache_size_gb    = 100
    enable_monitoring = true
  }
}

# Load Balancer with Cloud Armor
resource "google_compute_security_policy" "sie_x_policy" {
  name = "sie-x-prod-security-policy"

  rule {
    action   = "rate_based_ban"
    priority = 1000

    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"

      rate_limit_threshold {
        count        = 100
        interval_sec = 60
      }

      ban_duration_sec = 600
    }

    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
  }

  rule {
    action   = "allow"
    priority = 2147483647
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
  }
}

# CDN
resource "google_compute_backend_bucket" "static_assets" {
  name        = "sie-x-prod-static"
  bucket_name = google_storage_bucket.static_assets.name
  enable_cdn  = true

  cdn_policy {
    cache_key_policy {
      include_host         = true
      include_protocol     = true
      include_query_string = false
    }

    cache_mode                   = "CACHE_ALL_STATIC"
    client_ttl                   = 3600
    default_ttl                  = 3600
    max_ttl                      = 86400
    negative_caching             = true
    serve_while_stale            = 86400
  }
}

resource "google_storage_bucket" "static_assets" {
  name          = "sie-x-prod-static-assets"
  location      = "EU"
  storage_class = "MULTI_REGIONAL"

  cors {
    origin          = ["https://sie-x.example.com"]
    method          = ["GET", "HEAD"]
    response_header = ["*"]
    max_age_seconds = 3600
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
}

# Monitoring Alerts
resource "google_monitoring_alert_policy" "api_latency" {
  display_name = "SIE-X API Latency"
  combiner     = "OR"

  conditions {
    display_name = "API response time > 500ms"

    condition_threshold {
      filter          = "metric.type=\"kubernetes.io/ingress/request_latencies\" resource.type=\"k8s_ingress\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 500

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_PERCENTILE_99"
      }
    }
  }

  notification_channels = [
    google_monitoring_notification_channel.pagerduty.name,
    google_monitoring_notification_channel.slack.name
  ]
}

resource "google_monitoring_notification_channel" "pagerduty" {
  display_name = "PagerDuty"
  type         = "pagerduty"

  sensitive_labels {
    service_key = var.pagerduty_key
  }
}

resource "google_monitoring_notification_channel" "slack" {
  display_name = "Slack #sie-x-alerts"
  type         = "slack"

  labels = {
    channel_name = "#sie-x-alerts"
  }

  sensitive_labels {
    url = var.slack_webhook_url
  }
}