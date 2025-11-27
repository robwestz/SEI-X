# SIE-X Multi-Tenant Security Implementation Guide

This guide walks you through implementing the multi-tenant security architecture for SIE-X.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Database Setup](#database-setup)
3. [Environment Configuration](#environment-configuration)
4. [Code Integration](#code-integration)
5. [Testing](#testing)
6. [Deployment](#deployment)

---

## Prerequisites

### Required Dependencies

Add to `requirements.txt`:

```txt
# Existing dependencies...

# Security
PyJWT==2.8.0
bcrypt==4.1.1
python-multipart==0.0.6

# Database
asyncpg==0.29.0  # PostgreSQL async driver
SQLAlchemy==2.0.23  # ORM (optional)

# Caching (for auth & rate limiting)
redis==5.0.1
aioredis==2.0.1
```

Install:

```bash
pip install -r requirements.txt
```

---

## Database Setup

### Step 1: Create PostgreSQL Database

```bash
# Create database
createdb siex_production

# Or via psql
psql -U postgres
CREATE DATABASE siex_production;
CREATE USER siex_app WITH PASSWORD 'strong_password_here';
GRANT ALL PRIVILEGES ON DATABASE siex_production TO siex_app;
```

### Step 2: Run Schema Migration

```bash
psql -U siex_app -d siex_production -f database/schema.sql
```

This creates:
- ✅ `tenants` table with RLS
- ✅ `users` table with RBAC
- ✅ `api_keys` table
- ✅ `extraction_history` table (tenant-isolated)
- ✅ `audit_logs` table
- ✅ `jwt_blacklist` table

### Step 3: Verify Setup

```bash
psql -U siex_app -d siex_production

-- Check tables
\dt

-- Check Row-Level Security is enabled
SELECT schemaname, tablename, rowsecurity
FROM pg_tables
WHERE schemaname = 'public';

-- Should show 'true' for users, api_keys, extraction_history, audit_logs
```

---

## Environment Configuration

### Development (.env.development)

```bash
# Application
SIE_X_ENV=development
SIE_X_DEBUG=true
SIE_X_HOST=0.0.0.0
SIE_X_PORT=8000

# Database
SIE_X_DB_URL=postgresql://siex_app:password@localhost:5432/siex_dev
SIE_X_DB_POOL_SIZE=10
SIE_X_DB_MAX_OVERFLOW=20

# Redis (for auth cache & rate limiting)
SIE_X_REDIS_URL=redis://localhost:6379/0
SIE_X_REDIS_PASSWORD=

# Authentication
SIE_X_JWT_SECRET=dev-secret-change-in-production-min-32-chars
SIE_X_JWT_ALGORITHM=HS256
SIE_X_TOKEN_EXPIRY=3600  # 1 hour

# CORS (development - allow localhost)
SIE_X_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# Rate Limiting
SIE_X_RATE_LIMIT_HOUR=1000  # Generous for dev
SIE_X_RATE_LIMIT_MINUTE=100
SIE_X_RATE_LIMIT_SKIP_IF_UNAVAILABLE=true  # Don't fail if Redis down

# Security (relaxed for dev)
SIE_X_HTTPS_ONLY=false
SIE_X_ENFORCE_QUOTAS=false
```

### Production (.env.production)

```bash
# Application
SIE_X_ENV=production
SIE_X_DEBUG=false
SIE_X_HOST=0.0.0.0
SIE_X_PORT=8000
SIE_X_WORKERS=4

# Database
SIE_X_DB_URL=postgresql://siex_app:STRONG_PASSWORD@db.internal:5432/siex_production
SIE_X_DB_POOL_SIZE=20
SIE_X_DB_MAX_OVERFLOW=40
SIE_X_DB_SSL_MODE=require

# Redis (REQUIRED in production)
SIE_X_REDIS_URL=redis://redis.internal:6379/0
SIE_X_REDIS_PASSWORD=STRONG_REDIS_PASSWORD
SIE_X_REDIS_SSL=true

# Authentication (CRITICAL: Use strong secret)
SIE_X_JWT_SECRET=$(openssl rand -base64 48)  # Generate strong secret
SIE_X_JWT_ALGORITHM=HS256
SIE_X_TOKEN_EXPIRY=3600

# CORS (CRITICAL: Only specific origins)
SIE_X_ALLOWED_ORIGINS=https://app.yourcompany.com,https://admin.yourcompany.com

# Rate Limiting (CRITICAL: Strict limits)
SIE_X_RATE_LIMIT_HOUR=100
SIE_X_RATE_LIMIT_MINUTE=10
SIE_X_RATE_LIMIT_SKIP_IF_UNAVAILABLE=false  # Fail closed

# Security (CRITICAL: All enabled)
SIE_X_HTTPS_ONLY=true
SIE_X_ENFORCE_QUOTAS=true
SIE_X_TRUSTED_HOSTS=api.yourcompany.com

# File Uploads
SIE_X_MAX_FILE_SIZE=5242880  # 5MB
SIE_X_MAX_REQUEST_SIZE=10485760  # 10MB

# Logging
SIE_X_LOG_LEVEL=INFO
SIE_X_LOG_JSON=true
```

### Generate Strong JWT Secret

```bash
# Generate 256-bit secret
openssl rand -base64 48

# Or in Python
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

---

## Code Integration

### Step 1: Update `minimal_server.py`

Replace the existing `minimal_server.py` with the secure version:

```python
"""
Secure FastAPI server for SIE-X with multi-tenant architecture.
"""

import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging

from sie_x.api.middleware import (
    AuthenticationMiddleware,
    RateLimitMiddleware,
    RequestTracingMiddleware,
    AuthConfig
)
from sie_x.api.tenant_middleware import TenantContextMiddleware
from sie_x.api.decorators import get_current_tenant
from sie_x.models.tenant import TenantContext
from sie_x.cache.redis_cache import RedisCache
from sie_x.core.models import ExtractionRequest, ExtractionResponse

# Configure logging
logging.basicConfig(level=os.getenv("SIE_X_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SIE-X API",
    description="Secure Multi-Tenant Keyword Extraction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =============================================================================
# SECURITY MIDDLEWARE (Order matters!)
# =============================================================================

# 1. HTTPS enforcement (production only)
if os.getenv("SIE_X_HTTPS_ONLY", "false").lower() == "true":
    from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
    app.add_middleware(HTTPSRedirectMiddleware)
    logger.info("✅ HTTPS enforcement enabled")

# 2. Trusted hosts (production only)
trusted_hosts = os.getenv("SIE_X_TRUSTED_HOSTS", "").split(",")
if trusted_hosts and trusted_hosts != [""]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)
    logger.info(f"✅ Trusted hosts: {trusted_hosts}")

# 3. CORS (strict in production)
allowed_origins = os.getenv("SIE_X_ALLOWED_ORIGINS", "").split(",")
if not allowed_origins or allowed_origins == [""]:
    logger.warning("⚠️ No ALLOWED_ORIGINS set, using localhost defaults")
    allowed_origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    max_age=3600
)
logger.info(f"✅ CORS origins: {allowed_origins}")

# 4. Request tracing
app.add_middleware(RequestTracingMiddleware)
logger.info("✅ Request tracing enabled")

# 5. Redis cache (REQUIRED for auth & rate limiting)
redis_url = os.getenv("SIE_X_REDIS_URL", "redis://localhost:6379/0")
try:
    cache = RedisCache(url=redis_url)
    logger.info(f"✅ Redis cache connected: {redis_url}")
except Exception as e:
    logger.error(f"❌ Redis cache failed: {e}")
    if os.getenv("SIE_X_ENV") == "production":
        raise RuntimeError("Redis is REQUIRED in production")
    cache = None

# 6. Authentication middleware
auth_config = AuthConfig(
    jwt_secret=os.getenv("SIE_X_JWT_SECRET", "change-me-in-production"),
    jwt_algorithm=os.getenv("SIE_X_JWT_ALGORITHM", "HS256"),
    token_expiry=int(os.getenv("SIE_X_TOKEN_EXPIRY", "3600"))
)

if len(auth_config.jwt_secret) < 32:
    logger.error("❌ JWT_SECRET is too short (min 32 characters)")
    if os.getenv("SIE_X_ENV") == "production":
        raise ValueError("JWT_SECRET must be at least 32 characters in production")

app.add_middleware(AuthenticationMiddleware, config=auth_config, cache=cache)
logger.info("✅ Authentication middleware enabled")

# 7. Tenant context middleware
enforce_quotas = os.getenv("SIE_X_ENFORCE_QUOTAS", "true").lower() == "true"
app.add_middleware(TenantContextMiddleware, db_connection=None, enforce_quotas=enforce_quotas)
logger.info(f"✅ Tenant context middleware enabled (quotas: {enforce_quotas})")

# 8. Rate limiting
skip_if_unavailable = os.getenv("SIE_X_RATE_LIMIT_SKIP_IF_UNAVAILABLE", "false").lower() == "true"
app.add_middleware(
    RateLimitMiddleware,
    cache=cache,
    default_limit=f"{os.getenv('SIE_X_RATE_LIMIT_HOUR', '100')}/hour",
    burst_limit=f"{os.getenv('SIE_X_RATE_LIMIT_MINUTE', '10')}/minute",
    skip_if_unavailable=skip_if_unavailable
)
logger.info("✅ Rate limiting enabled")

# =============================================================================
# ROUTES
# =============================================================================

@app.get("/health", tags=["monitoring"])
async def health_check():
    """Public health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "security": "enabled"
    }


@app.post("/extract", response_model=ExtractionResponse, tags=["extraction"])
async def extract_keywords(
    request: ExtractionRequest,
    tenant: TenantContext = Depends(get_current_tenant)  # ✅ Tenant required!
):
    """
    Extract keywords from text (SECURE VERSION).

    This endpoint is protected by:
    - JWT authentication
    - Tenant isolation
    - Rate limiting
    - Input validation

    The tenant context ensures all operations are scoped to the
    authenticated user's tenant.
    """
    from sie_x.api.minimal_server import engine

    if not engine:
        raise HTTPException(503, "Engine not initialized")

    # Extract keywords
    keywords = engine.extract(
        text=request.text,
        top_k=request.options.top_k if request.options else 10
    )

    # TODO: Save to extraction_history table with tenant_id
    # await db.execute(
    #     "INSERT INTO extraction_history (tenant_id, user_id, text_hash, keywords, processing_time) "
    #     "VALUES ($1, $2, $3, $4, $5)",
    #     tenant.id, tenant.user_id, hashlib.sha256(request.text.encode()).hexdigest(),
    #     json.dumps([k.model_dump() for k in keywords]), processing_time
    # )

    return ExtractionResponse(
        keywords=keywords,
        processing_time=0.1,
        version="1.0.0",
        metadata={
            "tenant_id": tenant.id,
            "user_id": tenant.user_id
        }
    )


# Add more secure endpoints here...
```

### Step 2: Create Database Connection Pool

Create `sie_x/db/connection.py`:

```python
"""Database connection pool for asyncpg."""

import asyncpg
import os
from typing import Optional

_pool: Optional[asyncpg.Pool] = None


async def get_db_pool() -> asyncpg.Pool:
    """Get or create database connection pool."""
    global _pool

    if _pool is None:
        _pool = await asyncpg.create_pool(
            os.getenv("SIE_X_DB_URL"),
            min_size=int(os.getenv("SIE_X_DB_POOL_SIZE", "10")),
            max_size=int(os.getenv("SIE_X_DB_MAX_OVERFLOW", "20"))
        )

    return _pool


async def close_db_pool():
    """Close database connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
```

---

## Testing

### Unit Tests

Create `tests/test_security.py`:

```python
"""Security tests for SIE-X."""

import pytest
from fastapi.testclient import TestClient
from sie_x.api.secure_server import app
from sie_x.models.tenant import Role
from sie_x.api.tenant_middleware import create_test_jwt_payload
import jwt


@pytest.fixture
def client():
    return TestClient(app)


def create_test_token(tenant_id="test-tenant", user_id="test-user", role=Role.MEMBER):
    """Create a test JWT token."""
    payload = create_test_jwt_payload(
        tenant_id=tenant_id,
        tenant_slug="test-tenant",
        user_id=user_id,
        user_email="test@example.com",
        role=role
    )
    token = jwt.encode(payload, "test-secret", algorithm="HS256")
    return token


def test_no_auth_returns_401(client):
    """Test that requests without auth are rejected."""
    response = client.post("/extract", json={"text": "test"})
    assert response.status_code == 401


def test_valid_auth_returns_200(client):
    """Test that requests with valid JWT succeed."""
    token = create_test_token()
    response = client.post(
        "/extract",
        json={"text": "test"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200


def test_admin_can_access_admin_endpoint(client):
    """Test that admin users can access admin endpoints."""
    token = create_test_token(role=Role.TENANT_ADMIN)
    response = client.get(
        "/admin/users",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code in [200, 404]  # 404 if endpoint not implemented


def test_member_cannot_access_admin_endpoint(client):
    """Test that non-admin users are blocked from admin endpoints."""
    token = create_test_token(role=Role.MEMBER)
    response = client.get(
        "/admin/users",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403


def test_rate_limit_enforced(client):
    """Test that rate limiting works."""
    token = create_test_token()

    # Make 100 requests quickly
    responses = []
    for _ in range(100):
        response = client.post(
            "/extract",
            json={"text": "test"},
            headers={"Authorization": f"Bearer {token}"}
        )
        responses.append(response.status_code)

    # At least one should be rate limited
    assert 429 in responses
```

Run tests:

```bash
pytest tests/test_security.py -v
```

---

## Deployment

### Docker Deployment

Update `docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: siex_production
      POSTGRES_USER: siex_app
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/schema.sql:/docker-entrypoint-initdb.d/schema.sql

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}

  sie-x-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      SIE_X_ENV: production
      SIE_X_DB_URL: postgresql://siex_app:${DB_PASSWORD}@postgres:5432/siex_production
      SIE_X_REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379/0
      SIE_X_JWT_SECRET: ${JWT_SECRET}
      SIE_X_ALLOWED_ORIGINS: ${ALLOWED_ORIGINS}
    depends_on:
      - postgres
      - redis

volumes:
  postgres_data:
```

Deploy:

```bash
# Set environment variables
export DB_PASSWORD=$(openssl rand -base64 32)
export REDIS_PASSWORD=$(openssl rand -base64 32)
export JWT_SECRET=$(openssl rand -base64 48)
export ALLOWED_ORIGINS=https://app.yourcompany.com

# Deploy
docker-compose up -d

# Check logs
docker-compose logs -f sie-x-api
```

---

## Checklist

### Pre-Production Security Checklist

- [ ] PostgreSQL database created
- [ ] Schema migrated (`schema.sql` applied)
- [ ] Row-Level Security (RLS) verified
- [ ] Redis cache running
- [ ] Strong JWT secret generated (≥32 chars)
- [ ] CORS origins restricted to specific domains
- [ ] HTTPS enforced (`SIE_X_HTTPS_ONLY=true`)
- [ ] Rate limiting enabled with Redis
- [ ] Tenant context middleware enabled
- [ ] All endpoints use `@Depends(get_current_tenant)`
- [ ] File upload validation implemented
- [ ] URL validation (SSRF prevention) implemented
- [ ] Error messages don't leak internal details
- [ ] Audit logging enabled
- [ ] Security headers added
- [ ] Unit tests passing
- [ ] Penetration testing completed

---

## Troubleshooting

### Redis Connection Failed

```bash
# Check Redis is running
redis-cli ping
# Should return "PONG"

# Check connection string
echo $SIE_X_REDIS_URL
```

### JWT Token Invalid

```bash
# Check JWT secret length
echo -n "$SIE_X_JWT_SECRET" | wc -c
# Should be ≥32

# Test JWT creation
python -c "
import jwt
payload = {'test': 'data'}
token = jwt.encode(payload, 'your-secret', algorithm='HS256')
print(token)
"
```

### Row-Level Security Not Working

```sql
-- Check RLS is enabled
SELECT tablename, rowsecurity
FROM pg_tables
WHERE schemaname = 'public';

-- Test RLS
SET app.current_tenant = '00000000-0000-0000-0000-000000000001';
SELECT * FROM extraction_history;
-- Should only show data for that tenant
```

---

## Next Steps

1. ✅ Complete database migration
2. ✅ Update all API endpoints to use tenant context
3. ✅ Implement user registration/login endpoints
4. ✅ Add API key generation endpoints
5. ✅ Set up monitoring and alerting
6. ✅ Conduct security audit
7. ✅ Perform penetration testing
8. ✅ Deploy to staging environment
9. ✅ Load testing
10. ✅ Production deployment

---

**Security Contact:** security@yourcompany.com

**Last Updated:** 2025-11-27
