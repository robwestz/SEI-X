"""Tests for the unified configuration system."""

import os
import pytest
from unittest.mock import patch

from sie_x.config import (
    SIEXConfig,
    EngineConfig,
    CacheConfig,
    AuthConfig,
    APIConfig,
    get_config,
    reset_config,
)


class TestDefaults:
    """Test that default config values are correct."""

    def test_engine_defaults(self):
        cfg = EngineConfig()
        assert cfg.mode == "balanced"
        assert cfg.simple_spacy_model == "en_core_web_sm"
        assert cfg.batch_size == 32
        assert cfg.similarity_threshold == 0.3
        assert cfg.min_confidence == 0.3
        assert cfg.pagerank_weight == 0.7
        assert cfg.frequency_weight == 0.3

    def test_cache_defaults(self):
        cfg = CacheConfig()
        assert cfg.redis_url == "redis://localhost:6379"
        assert cfg.default_ttl == 3600
        assert cfg.key_prefix == "siex:"
        assert cfg.max_size == 10000

    def test_api_defaults(self):
        cfg = APIConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.title == "SIE-X API"

    def test_auth_defaults(self):
        cfg = AuthConfig()
        assert cfg.algorithm == "HS256"
        assert cfg.access_token_expire_minutes == 30

    def test_root_config_defaults(self):
        cfg = SIEXConfig()
        assert cfg.env == "development"
        assert cfg.debug is False
        assert isinstance(cfg.engine, EngineConfig)
        assert isinstance(cfg.cache, CacheConfig)
        assert isinstance(cfg.auth, AuthConfig)


class TestEnvOverride:
    """Test environment variable overrides."""

    def test_engine_mode_override(self):
        with patch.dict(os.environ, {"SIEX_ENGINE__MODE": "fast"}):
            cfg = EngineConfig()
            assert cfg.mode == "fast"

    def test_cache_max_size_override(self):
        with patch.dict(os.environ, {"SIEX_CACHE__MAX_SIZE": "500"}):
            cfg = CacheConfig()
            assert cfg.max_size == 500

    def test_api_port_override(self):
        with patch.dict(os.environ, {"SIEX_API__PORT": "9000"}):
            cfg = APIConfig()
            assert cfg.port == 9000


class TestSingleton:
    """Test get_config singleton behaviour."""

    def test_get_config_returns_same_instance(self):
        reset_config()
        a = get_config()
        b = get_config()
        assert a is b

    def test_reset_config_clears_singleton(self):
        reset_config()
        a = get_config()
        reset_config()
        b = get_config()
        assert a is not b


class TestSecretStr:
    """Test that secret_key is masked."""

    def test_secret_key_not_in_repr(self):
        cfg = AuthConfig()
        r = repr(cfg)
        # SecretStr should hide the value
        assert "dev_secret_key_change_in_production" not in r

    def test_secret_key_get_value(self):
        cfg = AuthConfig()
        assert cfg.secret_key.get_secret_value() == "dev_secret_key_change_in_production"


class TestNestedTypes:
    """Test that nested config objects have the right types."""

    def test_nested_types(self):
        cfg = SIEXConfig()
        assert isinstance(cfg.engine.batch_size, int)
        assert isinstance(cfg.engine.similarity_threshold, float)
        assert isinstance(cfg.cache.redis_url, str)
        assert isinstance(cfg.auth.access_token_expire_minutes, int)
        assert isinstance(cfg.api.cors_origins, list)
