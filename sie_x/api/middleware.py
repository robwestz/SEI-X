"""
API middleware for auth, rate limiting, and request tracking.

Uses FallbackCache for Redis/Memcached with graceful degradation.
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
from dataclasses import dataclass
import hashlib
import structlog

from sie_x.cache.redis_cache import FallbackCache, CacheBackend

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
    """JWT and API key authentication middleware with fallback cache support."""

    def __init__(self, app, config: AuthConfig, cache: CacheBackend):
        super().__init__(app)
        self.config = config
        self.cache = cache

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
        """Verify JWT token with cache support."""
        # Check if token is blacklisted (graceful degradation if cache unavailable)
        if self.cache and self.cache.is_available:
            blacklisted = await self.cache.get(f"blacklist:{token}")
            if blacklisted:
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
        """Verify API key against cache (Redis/Memcached)."""
        if not self.cache or not self.cache.is_available:
            raise HTTPException(
                status_code=503,
                detail="Authentication cache unavailable"
            )

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        user_data = await self.cache.get(f"api_key:{key_hash}")

        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return user_data  # Already deserialized by cache


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting with cache backend.

    Note: Requires Redis for counter operations (incr/expire).
    Gracefully skips rate limiting if cache unavailable (dev mode).
    """

    def __init__(
            self,
            app,
            cache: CacheBackend,
            default_limit: str = "100/hour",
            burst_limit: str = "10/minute",
            skip_if_unavailable: bool = True
    ):
        super().__init__(app)
        self.cache = cache
        self.default_limit = default_limit
        self.burst_limit = burst_limit
        self.skip_if_unavailable = skip_if_unavailable

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting if cache unavailable (dev mode)
        if not self.cache or not self.cache.is_available:
            if self.skip_if_unavailable:
                logger.warning("Rate limiting skipped: cache unavailable")
                return await call_next(request)
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Rate limiting unavailable"
                )

        # Get user identifier
        user_id = getattr(request.state, 'user', {}).get('id', 'anonymous')
        client_ip = request.client.host

        # Create rate limit keys
        hour_key = f"rate_limit:hour:{user_id}:{client_ip}"
        minute_key = f"rate_limit:minute:{user_id}:{client_ip}"

        try:
            # NOTE: This uses Redis-specific commands (incr/expire)
            # FallbackCache needs a Redis backend for rate limiting
            # TODO: Implement counter support in cache abstraction
            from sie_x.cache.redis_cache import RedisCache

            if isinstance(self.cache, RedisCache) and self.cache.client:
                redis_client = self.cache.client

                # Check rate limits
                hour_count = await redis_client.incr(hour_key)
                if hour_count == 1:
                    await redis_client.expire(hour_key, 3600)

                minute_count = await redis_client.incr(minute_key)
                if minute_count == 1:
                    await redis_client.expire(minute_key, 60)

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
            else:
                # Non-Redis backend - skip rate limiting
                logger.warning("Rate limiting requires Redis backend, skipping")
                return await call_next(request)

        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            if self.skip_if_unavailable:
                return await call_next(request)
            else:
                raise


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