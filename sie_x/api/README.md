# SIE-X API Module

**FastAPI-based REST API for semantic keyword extraction**

The API module provides HTTP endpoints for accessing SIE-X's keyword extraction capabilities, including real-time extraction, batch processing, URL analysis, and streaming support.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Middleware](#middleware)
- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)
- [Configuration](#configuration)
- [Production Deployment](#production-deployment)

---

## Overview

### What is the API Module?

The API module exposes SIE-X functionality through a production-ready REST API built with FastAPI. It includes:

- **Core Extraction**: Extract keywords from text, URLs, and files
- **Streaming Support**: Real-time extraction for large documents
- **Batch Processing**: Process multiple documents in parallel
- **Multi-language**: Automatic language detection and extraction
- **Middleware**: Auth, rate limiting, request tracing
- **Caching**: Redis/Memcached integration for performance

### Module Structure

```
sie_x/api/
├── minimal_server.py    # Main FastAPI app with core endpoints
├── routes.py            # Additional routes (URL, file, search)
├── middleware.py        # Auth, rate limiting, tracing middleware
├── server.py            # Production server configuration
└── README.md           # This file
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn httpx beautifulsoup4
```

### 2. Start the Server

```bash
# Development mode (auto-reload)
uvicorn sie_x.api.minimal_server:app --reload --port 8000

# Production mode
uvicorn sie_x.api.minimal_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Test the API

```bash
# Check health
curl http://localhost:8000/health

# Extract keywords
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple announced new iPhone with advanced AI features"}'
```

### 4. View API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## API Endpoints

### Core Extraction

#### `POST /extract`

Extract keywords from text input.

**Request:**
```json
{
  "text": "Your text here",
  "top_k": 10,
  "min_confidence": 0.3,
  "language": "auto"
}
```

**Response:**
```json
{
  "keywords": [
    {
      "text": "keyword",
      "score": 0.95,
      "type": "ENTITY",
      "count": 3,
      "confidence": 0.92
    }
  ],
  "processing_time": 0.124,
  "version": "1.0.0",
  "metadata": {
    "text_length": 150,
    "language": "en"
  }
}
```

**Example:**
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/extract",
        json={
            "text": "OpenAI releases GPT-4 with improved reasoning",
            "top_k": 5
        }
    )
    keywords = response.json()["keywords"]
    for kw in keywords:
        print(f"{kw['text']}: {kw['score']:.2f}")
```

---

### Batch Processing

#### `POST /batch`

Process multiple documents in parallel.

**Request:**
```json
{
  "documents": [
    {"text": "First document..."},
    {"text": "Second document..."}
  ],
  "top_k": 10
}
```

**Response:**
```json
{
  "results": [
    {
      "keywords": [...],
      "processing_time": 0.15
    },
    {
      "keywords": [...],
      "processing_time": 0.12
    }
  ],
  "total_processing_time": 0.27
}
```

**Example:**
```python
documents = [
    "Article about climate change...",
    "Product review for new laptop...",
    "Breaking news about technology..."
]

response = await client.post(
    "http://localhost:8000/batch",
    json={
        "documents": [{"text": doc} for doc in documents],
        "top_k": 5
    }
)
```

---

### Streaming Extraction

#### `POST /stream`

Stream keyword extraction for large documents (>10K words).

**Request:**
```json
{
  "text": "Very long document...",
  "chunk_size": 1000,
  "strategy": "smart"
}
```

**Response:** Server-Sent Events (SSE) stream

```
data: {"chunk_id": 0, "keywords": [...], "progress": 25}
data: {"chunk_id": 1, "keywords": [...], "progress": 50}
data: {"chunk_id": 2, "keywords": [...], "progress": 75}
data: {"chunk_id": 3, "keywords": [...], "progress": 100, "is_final": true}
```

**Example:**
```python
async with client.stream(
    "POST",
    "http://localhost:8000/stream",
    json={"text": long_document, "chunk_size": 1000}
) as response:
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            data = json.loads(line[6:])
            print(f"Progress: {data['progress']}%")
            print(f"Keywords: {len(data['keywords'])}")
```

---

### URL Analysis

#### `POST /api/v1/analyze/url`

Fetch and analyze content from a URL.

**Request:**
```json
{
  "url": "https://example.com/article",
  "top_k": 10,
  "min_confidence": 0.3
}
```

**Response:** Same format as `/extract`

**Example:**
```python
response = await client.post(
    "http://localhost:8000/api/v1/analyze/url",
    params={
        "url": "https://techcrunch.com/2024/01/15/ai-breakthrough",
        "top_k": 10
    }
)
```

---

### File Upload

#### `POST /api/v1/analyze/file`

Upload and analyze a text file.

**Request:** `multipart/form-data`

**Example:**
```python
files = {"file": open("document.txt", "rb")}
response = await client.post(
    "http://localhost:8000/api/v1/analyze/file",
    files=files,
    params={"top_k": 10}
)
```

---

### Keyword Search

#### `GET /api/v1/search`

Search previously extracted keywords.

**Request:**
```
GET /api/v1/search?query=technology&limit=10
```

**Response:**
```json
{
  "results": [
    {
      "keyword": "artificial intelligence",
      "documents": ["doc1", "doc2"],
      "total_occurrences": 15
    }
  ]
}
```

---

### Health & Metrics

#### `GET /health`

Server health check.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "engines": {
    "simple_engine": true,
    "streaming_extractor": true,
    "multilang_engine": false
  }
}
```

#### `GET /metrics`

API usage metrics.

**Response:**
```json
{
  "total_extractions": 1543,
  "avg_processing_time": 0.156,
  "errors": 12,
  "uptime_hours": 24.5
}
```

---

## Middleware

### Authentication Middleware

JWT and API key authentication with cache-backed token validation.

**Features:**
- JWT token verification
- API key validation (SHA256 hashed)
- Token blacklist support
- Graceful cache fallback

**Usage:**
```python
from sie_x.api.middleware import AuthenticationMiddleware, AuthConfig
from sie_x.cache.redis_cache import FallbackCache

# Configure auth
auth_config = AuthConfig(
    jwt_secret="your-secret-key",
    jwt_algorithm="HS256",
    token_expiry=3600
)

# Create cache
cache = FallbackCache(redis_url="redis://localhost:6379")
await cache.connect()

# Add middleware
app.add_middleware(AuthenticationMiddleware, config=auth_config, cache=cache)
```

**Request Example:**
```bash
# JWT authentication
curl -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..." \
  http://localhost:8000/extract

# API key authentication
curl -H "Authorization: ApiKey sk_live_abc123..." \
  http://localhost:8000/extract
```

---

### Rate Limiting Middleware

Distributed rate limiting with Redis/Memcached backend.

**Features:**
- Hourly rate limits (default: 100/hour)
- Burst protection (default: 10/minute)
- Per-user and per-IP tracking
- Graceful degradation if cache unavailable

**Usage:**
```python
from sie_x.api.middleware import RateLimitMiddleware

app.add_middleware(
    RateLimitMiddleware,
    cache=cache,
    default_limit="100/hour",
    burst_limit="10/minute",
    skip_if_unavailable=True  # Dev mode: skip if Redis down
)
```

**Response Headers:**
```
X-RateLimit-Limit-Hour: 100
X-RateLimit-Remaining-Hour: 87
X-RateLimit-Limit-Minute: 10
X-RateLimit-Remaining-Minute: 9
```

**Rate Limit Exceeded:**
```json
{
  "detail": "Rate limit exceeded: 101/100 per hour"
}
```

---

### Request Tracing Middleware

Distributed tracing and request correlation.

**Features:**
- Automatic request ID generation
- Request/response logging
- Processing time tracking
- Structured logging with context

**Usage:**
```python
from sie_x.api.middleware import RequestTracingMiddleware

app.add_middleware(RequestTracingMiddleware)
```

**Response Headers:**
```
X-Request-ID: req_1732094000123456
X-Response-Time: 0.156s
```

**Logs:**
```
INFO: request_started method=POST path=/extract client_ip=192.168.1.10 request_id=req_...
INFO: request_completed method=POST status=200 duration=0.156 request_id=req_...
```

---

## Configuration

### Environment Variables

```bash
# Server
export SIE_X_HOST="0.0.0.0"
export SIE_X_PORT="8000"
export SIE_X_WORKERS="4"

# Cache
export SIE_X_REDIS_URL="redis://localhost:6379"
export SIE_X_MEMCACHED_SERVERS="localhost:11211"

# Authentication
export SIE_X_JWT_SECRET="your-secret-key"
export SIE_X_JWT_ALGORITHM="HS256"

# Rate Limiting
export SIE_X_RATE_LIMIT_HOUR="100"
export SIE_X_RATE_LIMIT_MINUTE="10"

# Logging
export SIE_X_LOG_LEVEL="INFO"
```

### Production Configuration

```python
# Production server with all middleware
from sie_x.api.minimal_server import app
from sie_x.api.middleware import (
    AuthenticationMiddleware,
    RateLimitMiddleware,
    RequestTracingMiddleware,
    AuthConfig
)
from sie_x.cache.redis_cache import FallbackCache
import os

# Initialize cache
cache = FallbackCache(
    redis_url=os.getenv("SIE_X_REDIS_URL", "redis://localhost:6379"),
    memcached_servers=[os.getenv("SIE_X_MEMCACHED_SERVERS", "localhost:11211")]
)

# Auth configuration
auth_config = AuthConfig(
    jwt_secret=os.getenv("SIE_X_JWT_SECRET"),
    jwt_algorithm=os.getenv("SIE_X_JWT_ALGORITHM", "HS256")
)

# Add middleware
app.add_middleware(RequestTracingMiddleware)
app.add_middleware(RateLimitMiddleware, cache=cache)
app.add_middleware(AuthenticationMiddleware, config=auth_config, cache=cache)

# Run with uvicorn
# uvicorn sie_x.api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sie_x/ ./sie_x/
COPY setup.py .
RUN pip install -e .

# Download spaCy models
RUN python -m spacy download en_core_web_sm

EXPOSE 8000
CMD ["uvicorn", "sie_x.api.minimal_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sie-x-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sie-x-api
  template:
    metadata:
      labels:
        app: sie-x-api
    spec:
      containers:
      - name: api
        image: sie-x:latest
        ports:
        - containerPort: 8000
        env:
        - name: SIE_X_REDIS_URL
          value: "redis://redis-service:6379"
        - name: SIE_X_JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: sie-x-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

### NGINX Reverse Proxy

```nginx
upstream sie_x_backend {
    least_conn;
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    server_name api.sie-x.com;

    location / {
        proxy_pass http://sie_x_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Timeouts for long-running requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
}
```

---

## Performance Optimization

### Caching Strategy

```python
from sie_x.cache.redis_cache import FallbackCache

cache = FallbackCache(
    redis_url="redis://localhost:6379",
    default_ttl=3600  # 1 hour
)

# Cache extraction results
cache_key = f"extract:{hash(text)}"
cached = await cache.get(cache_key)

if cached:
    return cached
else:
    result = engine.extract(text)
    await cache.set(cache_key, result, ttl=3600)
    return result
```

### Load Balancing

- Use multiple uvicorn workers: `--workers 4`
- Deploy across multiple servers
- Use Redis for shared cache/session state
- Implement connection pooling for database access

### Monitoring

```python
from sie_x.monitoring.metrics import PrometheusMetrics

metrics = PrometheusMetrics()

@app.middleware("http")
async def track_metrics(request, call_next):
    with metrics.track_request(request.url.path):
        response = await call_next(request)
    return response
```

---

## Troubleshooting

### Common Issues

**1. Engine not initialized**
```
{"detail": "Engine not initialized"}
```
**Fix:** Wait for server startup to complete (~5-10 seconds for model loading)

**2. Rate limit exceeded**
```
{"detail": "Rate limit exceeded: 101/100 per hour"}
```
**Fix:** Wait for rate limit window to reset, or increase limits in config

**3. Cache unavailable**
```
WARNING: Rate limiting skipped: cache unavailable
```
**Fix:** Ensure Redis/Memcached is running. In dev mode, this is expected and safe.

**4. Import errors**
```
ModuleNotFoundError: No module named 'sie_x'
```
**Fix:** Install package: `pip install -e .` from project root

---

## API Versioning

Current version: **v1.0.0**

**Version Strategy:**
- URL-based versioning: `/api/v1/...`
- Header-based versioning: `Accept-Version: 1.0`
- Semantic versioning for releases

**Breaking Changes:**
- Major version bumps only (v1 → v2)
- Maintain v1 for 6 months after v2 release
- Deprecation warnings 3 months before removal

---

## Security Best Practices

1. **Use HTTPS in production** - Never use HTTP for sensitive data
2. **Rotate JWT secrets** - Change secrets every 90 days
3. **Hash API keys** - Store SHA256 hashes, not plaintext
4. **Enable rate limiting** - Prevent abuse and DoS attacks
5. **Validate input** - Use Pydantic models for request validation
6. **Sanitize output** - Prevent XSS in returned data
7. **Monitor logs** - Track authentication failures and suspicious patterns

---

## Related Documentation

- [Core Engine](../core/README.md) - Keyword extraction engine
- [SDK Documentation](../sdk/README.md) - Python client library
- [Transformers](../transformers/README.md) - Domain-specific extractors
- [Cache Module](../cache/README.md) - Redis/Memcached caching
- [Monitoring](../monitoring/README.md) - Metrics and observability

---

## License

Part of the SIE-X project - Semantic Intelligence Engine X

---

**Last Updated:** 2025-11-23
**Module Version:** 1.0.0
**Maintainer:** SIE-X Team
