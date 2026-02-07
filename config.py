# -*- coding: utf-8 -*-
"""
SIE-X Unified Configuration Layer.

All configuration in one place. Values can be overridden via:
1. Environment variables (SIEX_ prefix, __ for nesting)
2. .env file (dev)
3. Secrets directory (Kubernetes /run/secrets)

Examples:
    SIEX_ENGINE__MODE=fast
    SIEX_CACHE__REDIS_URL=redis://prod-host:6379
    SIEX_AUTH__SECRET_KEY=my-production-secret
"""

from typing import Dict, List, Optional, Any
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EngineConfig(BaseSettings):
    """Core engine configuration."""
    model_config = SettingsConfigDict(env_prefix="SIEX_ENGINE__")

    mode: str = "balanced"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    simple_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    spacy_models: List[str] = ["en_core_web_lg", "en_core_web_md", "xx_ent_wiki_sm"]
    simple_spacy_model: str = "en_core_web_sm"
    embedding_dim: int = 768
    batch_size: int = 32
    max_chunk_size: int = 512
    cache_size: int = 10000
    enable_gpu: bool = True
    enable_monitoring: bool = True

    # Thresholds
    similarity_threshold: float = 0.3
    min_confidence: float = 0.3
    pagerank_alpha: float = 0.85
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 2
    chunking_threshold: int = 5000
    overlap_ratio: float = 0.1
    related_terms_top_n: int = 3

    # Scoring weights (simple engine)
    pagerank_weight: float = 0.7
    frequency_weight: float = 0.3

    # Fine-tuning defaults
    finetune_epochs: int = 3
    finetune_lr: float = 2e-5


class APIConfig(BaseSettings):
    """API server configuration."""
    model_config = SettingsConfigDict(env_prefix="SIEX_API__")

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["*"]
    title: str = "SIE-X API"
    version: str = "1.0.0"


class AuthConfig(BaseSettings):
    """Authentication configuration."""
    model_config = SettingsConfigDict(env_prefix="SIEX_AUTH__")

    secret_key: SecretStr = SecretStr("dev_secret_key_change_in_production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    token_expiry_seconds: int = 3600
    api_keys_enabled: bool = True


class CacheConfig(BaseSettings):
    """Cache configuration."""
    model_config = SettingsConfigDict(env_prefix="SIEX_CACHE__")

    redis_url: str = "redis://localhost:6379"
    memcached_servers: List[str] = ["localhost:11211"]
    default_ttl: int = 3600
    key_prefix: str = "siex:"
    max_size: int = 10000  # In-memory LRU cache


class StreamingConfig(BaseSettings):
    """Kafka streaming configuration."""
    model_config = SettingsConfigDict(env_prefix="SIEX_STREAMING__")

    kafka_brokers: List[str] = ["localhost:9092"]
    input_topic: str = "siex-input"
    output_topic: str = "siex-output"
    batch_size: int = 10
    batch_timeout: float = 1.0
    redis_url: str = "redis://localhost:6379"


class MonitoringConfig(BaseSettings):
    """Observability configuration."""
    model_config = SettingsConfigDict(env_prefix="SIEX_MONITORING__")

    otlp_endpoint: str = "localhost:4317"
    otlp_insecure: bool = True
    log_level: str = "INFO"


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration."""
    model_config = SettingsConfigDict(env_prefix="SIEX_RATE_LIMIT__")

    default_limit: str = "100/hour"
    burst_limit: str = "10/minute"
    skip_if_unavailable: bool = True


class ResilienceConfig(BaseSettings):
    """Retry and circuit breaker configuration."""
    model_config = SettingsConfigDict(env_prefix="SIEX_RESILIENCE__")

    max_retries: int = 3
    max_backoff_seconds: int = 300
    circuit_breaker_timeout: int = 60
    max_memory_mb: int = 4096
    max_concurrent: int = 100
    cleanup_interval: int = 300


class MultilingualConfig(BaseSettings):
    """Multilingual support configuration."""
    model_config = SettingsConfigDict(env_prefix="SIEX_MULTILINGUAL__")

    default_language: str = "en"
    spacy_models: Dict[str, str] = {
        "en": "en_core_web_sm",
        "sv": "sv_core_news_sm",
        "es": "es_core_news_sm",
        "fr": "fr_core_news_sm",
        "de": "de_core_news_sm",
        "it": "it_core_news_sm",
        "pt": "pt_core_news_sm",
        "nl": "nl_core_news_sm",
        "el": "el_core_news_sm",
        "nb": "nb_core_news_sm",
        "lt": "lt_core_news_sm",
    }


class SDKConfig(BaseSettings):
    """SDK client configuration."""
    model_config = SettingsConfigDict(env_prefix="SIEX_SDK__")

    base_url: str = "http://localhost:8000"
    timeout: float = 30.0
    max_retries: int = 3


class AutoMLConfig(BaseSettings):
    """AutoML optimizer configuration."""
    model_config = SettingsConfigDict(env_prefix="SIEX_AUTOML__")

    n_trials: int = 100
    n_jobs: int = -1
    objective_metric: str = "f1_score"


class SIEXConfig(BaseSettings):
    """
    Root configuration for SIE-X.

    All settings can be overridden via environment variables with SIEX_ prefix.
    Nested settings use __ separator: SIEX_ENGINE__MODE=fast

    Supports .env files in the project root for development.
    """
    model_config = SettingsConfigDict(
        env_prefix="SIEX_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Environment profile
    env: str = "development"
    debug: bool = False

    # Nested configs
    engine: EngineConfig = EngineConfig()
    api: APIConfig = APIConfig()
    auth: AuthConfig = AuthConfig()
    cache: CacheConfig = CacheConfig()
    streaming: StreamingConfig = StreamingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    resilience: ResilienceConfig = ResilienceConfig()
    multilingual: MultilingualConfig = MultilingualConfig()
    sdk: SDKConfig = SDKConfig()
    automl: AutoMLConfig = AutoMLConfig()

    # System-specific config (open dict for domain plugins)
    systems: Dict[str, Any] = {}


# --- Singleton ---

_config: Optional[SIEXConfig] = None


def get_config() -> SIEXConfig:
    """Get the global SIE-X configuration (lazy singleton)."""
    global _config
    if _config is None:
        _config = SIEXConfig()
    return _config


def reset_config() -> None:
    """Reset config (useful for testing)."""
    global _config
    _config = None
