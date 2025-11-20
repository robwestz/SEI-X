"""
API middleware for auth, rate limiting, and request tracking.
"""

from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
import json
import time
import logging
import jwt
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis.asyncio as redis
from dataclasses import dataclass
import hashlib
import structlog

# Set up logger
logger = structlog.get_logger()

# Security
security = HTTPBearer()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@dataclass
class AuthConfig:
    """Authentication configuration."""
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    token_expiry: int = 3600  # 1 hour
    api_keys_enabled: bool = True
    oauth_enabled: bool = True


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """JWT and API key authentication middleware."""

    def __init__(self, app, config: AuthConfig, redis_client: redis.Redis):
        super().__init__(app)
        self.config = config
        self.redis = redis_client

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health checks
        if request.url.path in ["/health", "/metrics", "/docs"]:
            return await call_next(request)

        # Check authentication
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing authentication")

        try:
            # Try JWT first
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                payload = await self._verify_jwt(token)
                request.state.user = payload

            # Try API key
            elif auth_header.startswith("ApiKey "):
                api_key = auth_header.split(" ")[1]
                user = await self._verify_api_key(api_key)
                request.state.user = user

            else:
                raise HTTPException(status_code=401, detail="Invalid auth scheme")

            # Process request
            response = await call_next(request)
            return response

        except jwt.PyJWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def _verify_jwt(self, token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        # Check if token is blacklisted
        if await self.redis.get(f"blacklist:{token}"):
            raise HTTPException(status_code=401, detail="Token revoked")

        payload = jwt.decode(
            token,
            self.config.jwt_secret,
            algorithms=[self.config.jwt_algorithm]
        )

        # Verify expiration
        if payload.get("exp", 0) < datetime.utcnow().timestamp():
            raise HTTPException(status_code=401, detail="Token expired")

        return payload

    async def _verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify API key against Redis."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        user_data = await self.redis.hget("api_keys", key_hash)

        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return json.loads(user_data)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting with Redis backend."""

    def __init__(
            self,
            app,
            redis_client: redis.Redis,
            default_limit: str = "100/hour",
            burst_limit: str = "10/minute"
    ):
        super().__init__(app)
        self.redis = redis_client
        self.default_limit = default_limit
        self.burst_limit = burst_limit

    async def dispatch(self, request: Request, call_next):
        # Get user identifier
        user_id = getattr(request.state, 'user', {}).get('id', 'anonymous')
        client_ip = request.client.host

        # Create rate limit keys
        hour_key = f"rate_limit:hour:{user_id}:{client_ip}"
        minute_key = f"rate_limit:minute:{user_id}:{client_ip}"

        # Check rate limits
        hour_count = await self.redis.incr(hour_key)
        if hour_count == 1:
            await self.redis.expire(hour_key, 3600)

        minute_count = await self.redis.incr(minute_key)
        if minute_count == 1:
            await self.redis.expire(minute_key, 60)

        # Parse limits
        hour_limit = int(self.default_limit.split('/')[0])
        minute_limit = int(self.burst_limit.split('/')[0])

        # Check if exceeded
        if hour_count > hour_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {hour_count}/{hour_limit} per hour"
            )

        if minute_count > minute_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Burst limit exceeded: {minute_count}/{minute_limit} per minute"
            )

        # Add headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit-Hour"] = str(hour_limit)
        response.headers["X-RateLimit-Remaining-Hour"] = str(hour_limit - hour_count)
        response.headers["X-RateLimit-Limit-Minute"] = str(minute_limit)
        response.headers["X-RateLimit-Remaining-Minute"] = str(minute_limit - minute_count)

        return response


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Request tracing and correlation."""

    async def dispatch(self, request: Request, call_next):
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", self._generate_request_id())
        request.state.request_id = request_id

        # Add to logs
        with logger.contextvars.bind(request_id=request_id):
            # Log request
            logger.info(
                "request_started",
                method=request.method,
                path=request.url.path,
                client_ip=request.client.host
            )

            # Process
            start_time = time.time()
            response = await call_next(request)
            duration = time.time() - start_time

            # Log response
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=duration
            )

            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"

            return response

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return f"req_{int(time.time() * 1000000)}"