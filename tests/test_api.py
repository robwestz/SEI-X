"""Tests for the FastAPI endpoints in minimal_server.py.

Uses TestClient with the engine mocked at startup.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from fastapi.testclient import TestClient


def _get_test_client():
    """Create a TestClient with all heavy deps mocked."""
    from sie_x.tests.conftest import MockEngine

    mock_engine = MockEngine()

    import sie_x.api.minimal_server as srv
    original_engine = srv.engine
    original_startup = srv.startup_time
    srv.engine = mock_engine
    srv.startup_time = datetime(2025, 1, 1)

    # Reset rate-limit store so tests don't interfere
    srv.rate_limit_store.clear()
    srv.stats["total_extractions"] = 0
    srv.stats["total_processing_time"] = 0.0
    srv.stats["errors"] = 0

    client = TestClient(srv.app, raise_server_exceptions=False)
    return client, srv, mock_engine, original_engine, original_startup


def _auth_header():
    """Generate a valid Bearer token directly (bypasses /token endpoint)."""
    from sie_x.api.auth import create_access_token
    token = create_access_token(
        data={"sub": "admin", "role": "admin"},
        expires_delta=timedelta(minutes=30),
    )
    return {"Authorization": f"Bearer {token}"}


def _api_key_header():
    """Use a valid API key for auth."""
    return {"X-API-Key": "siex-dev-key-123"}


# ============================================================================
# Tests
# ============================================================================


class TestRootEndpoint:

    def test_root_returns_200(self):
        client, srv, _, orig_eng, orig_st = _get_test_client()
        try:
            resp = client.get("/")
            assert resp.status_code == 200
            body = resp.json()
            assert "endpoints" in body
            assert "name" in body
        finally:
            srv.engine = orig_eng
            srv.startup_time = orig_st


class TestHealth:

    def test_health_returns_200(self):
        client, srv, _, orig_eng, orig_st = _get_test_client()
        try:
            resp = client.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "healthy"
            assert "models_loaded" in body
        finally:
            srv.engine = orig_eng
            srv.startup_time = orig_st


class TestToken:

    def test_valid_credentials(self):
        """Test /token with valid credentials. Patches the datetime.timedelta bug."""
        client, srv, _, orig_eng, orig_st = _get_test_client()
        try:
            with patch.object(srv, "datetime", wraps=None) as mock_dt:
                # The endpoint uses datetime.timedelta — patch it to work
                import datetime as dt_module
                mock_dt.timedelta = dt_module.timedelta
                mock_dt.now = datetime.now

                resp = client.post(
                    "/token",
                    data={"username": "admin", "password": "admin"},
                )
            assert resp.status_code == 200
            body = resp.json()
            assert "access_token" in body
            assert body["token_type"] == "bearer"
        finally:
            srv.engine = orig_eng
            srv.startup_time = orig_st

    def test_invalid_credentials(self):
        client, srv, _, orig_eng, orig_st = _get_test_client()
        try:
            resp = client.post(
                "/token",
                data={"username": "admin", "password": "wrong"},
            )
            assert resp.status_code == 401
        finally:
            srv.engine = orig_eng
            srv.startup_time = orig_st

    def test_unknown_user(self):
        client, srv, _, orig_eng, orig_st = _get_test_client()
        try:
            resp = client.post(
                "/token",
                data={"username": "nobody", "password": "x"},
            )
            assert resp.status_code == 401
        finally:
            srv.engine = orig_eng
            srv.startup_time = orig_st


class TestExtract:

    def test_extract_with_auth(self):
        client, srv, _, orig_eng, orig_st = _get_test_client()
        try:
            headers = _auth_header()
            resp = client.post(
                "/extract",
                json={"text": "Machine learning is great"},
                headers=headers,
            )
            assert resp.status_code == 200
            body = resp.json()
            assert "keywords" in body
            assert "processing_time" in body
        finally:
            srv.engine = orig_eng
            srv.startup_time = orig_st

    def test_extract_with_api_key(self):
        client, srv, _, orig_eng, orig_st = _get_test_client()
        try:
            headers = _api_key_header()
            resp = client.post(
                "/extract",
                json={"text": "Machine learning is great"},
                headers=headers,
            )
            assert resp.status_code == 200
        finally:
            srv.engine = orig_eng
            srv.startup_time = orig_st

    def test_extract_without_auth(self):
        client, srv, _, orig_eng, orig_st = _get_test_client()
        try:
            resp = client.post(
                "/extract",
                json={"text": "Machine learning is great"},
            )
            assert resp.status_code == 401
        finally:
            srv.engine = orig_eng
            srv.startup_time = orig_st


class TestExtractBatch:

    def test_batch_with_auth(self):
        client, srv, _, orig_eng, orig_st = _get_test_client()
        try:
            headers = _auth_header()
            resp = client.post(
                "/extract/batch",
                json={
                    "items": [
                        {"text": "First document about AI"},
                        {"text": "Second document about ML"},
                    ]
                },
                headers=headers,
            )
            assert resp.status_code == 200
            body = resp.json()
            assert isinstance(body, list)
            assert len(body) == 2
        finally:
            srv.engine = orig_eng
            srv.startup_time = orig_st


class TestStats:

    def test_stats_returns_200(self):
        client, srv, _, orig_eng, orig_st = _get_test_client()
        try:
            resp = client.get("/stats")
            assert resp.status_code == 200
            body = resp.json()
            assert "api_stats" in body
        finally:
            srv.engine = orig_eng
            srv.startup_time = orig_st
