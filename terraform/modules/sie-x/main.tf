# SIE-X Enterprise Terraform Module
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "europe-north1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "sie_x_config" {
  description = "SIE-X configuration"
  type = object({
    version          = string
    replicas         = number
    gpu_enabled      = bool
    gpu_type         = string
    max_gpu_per_node = number
    memory_gb        = number
    cpu_cores        = number
    models           = list(string)
    cache_size_gb    = number
    enable_monitoring = bool
  })

  default = {
    version          = "v3.0.0"
    replicas         = 3
    gpu_enabled      = true
    gpu_type         = "nvidia-tesla-t4"
    max_gpu_per_node = 2
    memory_gb        = 16
    cpu_cores        = 8
    models           = ["all-mpnet-base-v2", "LaBSE", "xlm-roberta-base"]
    cache_size_gb    = 50
    enable_monitoring = true
  }
}

# Network
resource "google_compute_network" "sie_x_vpc" {
  name                    = "sie-x-${var.environment}-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "sie_x_subnet" {
  name          = "sie-x-${var.environment}-subnet"
  network       = google_compute_network.sie_x_vpc.id
  ip_cidr_range = "10.0.0.0/20"
  region        = var.region

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/20"
  }

  private_ip_google_access = true
}

# GKE Cluster
resource "google_container_cluster" "sie_x_cluster" {
  name     = "sie-x-${var.environment}"
  location = var.region

  initial_node_count       = 1
  remove_default_node_pool = true

  network    = google_compute_network.sie_x_vpc.name
  subnetwork = google_compute_subnetwork.sie_x_subnet.name

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
    gcp_filestore_csi_driver_config {
      enabled = true
    }
    gcs_fuse_csi_driver_config {
      enabled = true
    }
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]

    managed_prometheus {
      enabled = true
    }
  }

  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }
}

# Node Pools
resource "google_container_node_pool" "sie_x_cpu_pool" {
  name       = "sie-x-cpu-pool"
  cluster    = google_container_cluster.sie_x_cluster.name
  location   = var.region
  node_count = 2

  autoscaling {
    min_node_count = 2
    max_node_count = 10
  }

  node_config {
    machine_type = "n2-standard-${var.sie_x_config.cpu_cores}"

    disk_size_gb = 100
    disk_type    = "pd-ssd"

    metadata = {
      disable-legacy-endpoints = "true"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
      node_type   = "cpu"
    }

    taint {
      key    = "sie-x-cpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }
  }
}

resource "google_container_node_pool" "sie_x_gpu_pool" {
  count = var.sie_x_config.gpu_enabled ? 1 : 0

  name       = "sie-x-gpu-pool"
  cluster    = google_container_cluster.sie_x_cluster.name
  location   = var.region
  node_count = 1

  autoscaling {
    min_node_count = 1
    max_node_count = 5
  }

  node_config {
    machine_type = "n1-standard-8"

    guest_accelerator {
      type  = var.sie_x_config.gpu_type
      count = var.sie_x_config.max_gpu_per_node

      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    disk_size_gb = 200
    disk_type    = "pd-ssd"

    metadata = {
      disable-legacy-endpoints = "true"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
      node_type   = "gpu"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }
  }
}

# Storage
resource "google_filestore_instance" "model_storage" {
  name     = "sie-x-${var.environment}-models"
  location = "${var.region}-a"
  tier     = "BASIC_SSD"

  file_shares {
    capacity_gb = 2560
    name        = "models"

    nfs_export_options {
      ip_ranges   = ["10.0.0.0/20"]
      access_mode = "READ_WRITE"
      squash_mode = "NO_ROOT_SQUASH"
    }
  }

  networks {
    network = google_compute_network.sie_x_vpc.name
    modes   = ["MODE_IPV4"]
  }
}

# Redis (Memorystore)
resource "google_redis_instance" "cache" {
  name           = "sie-x-${var.environment}-cache"
  tier           = var.environment == "prod" ? "STANDARD_HA" : "BASIC"
  memory_size_gb = var.sie_x_config.cache_size_gb
  region         = var.region

  redis_version = "REDIS_7_0"

  redis_configs = {
    maxmemory-policy = "allkeys-lru"
    notify-keyspace-events = "Ex"
  }

  auth_enabled = true

  persistence_config {
    persistence_mode    = "RDB"
    rdb_snapshot_period = "TWENTY_FOUR_HOURS"
  }
}

# Cloud SQL (PostgreSQL)
resource "google_sql_database_instance" "metadata_db" {
  name             = "sie-x-${var.environment}-metadata"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier = var.environment == "prod" ? "db-custom-8-32768" : "db-custom-2-8192"

    availability_type = var.environment == "prod" ? "REGIONAL" : "ZONAL"

    disk_type         = "PD_SSD"
    disk_size         = 100
    disk_autoresize   = true

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7

      backup_retention_settings {
        retained_backups = 30
        retention_unit   = "COUNT"
      }
    }

    database_flags {
      name  = "max_connections"
      value = "200"
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }
  }
}

# Service Accounts
resource "google_service_account" "sie_x_sa" {
  account_id   = "sie-x-${var.environment}"
  display_name = "SIE-X Service Account"
}

resource "google_project_iam_member" "sie_x_roles" {
  for_each = toset([
    "roles/storage.objectAdmin",
    "roles/cloudsql.client",
    "roles/redis.editor",
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter",
    "roles/cloudtrace.agent"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.sie_x_sa.email}"
}

# Workload Identity
resource "google_service_account_iam_member" "workload_identity" {
  service_account_id = google_service_account.sie_x_sa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[sie-x/sie-x-api]"
}

# Outputs
output "cluster_endpoint" {
  value = google_container_cluster.sie_x_cluster.endpoint
}

output "redis_host" {
  value = google_redis_instance.cache.host
}

output "postgres_connection" {
  value = google_sql_database_instance.metadata_db.connection_name
}

output "model_storage_ip" {
  value = google_filestore_instance.model_storage.networks[0].ip_addresses[0]
}