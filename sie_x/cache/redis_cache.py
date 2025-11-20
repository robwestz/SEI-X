"""
Distributed caching layer for SIE-X production deployment.

Provides distributed caching with multiple backend support:
- Redis (primary, high performance)
- Memcached (fallback, widely available)
- Automatic fallback on connection failure

Provides distributed caching for:
- Keyword extraction results
- Semantic embeddings
- SEO analysis results
- BACOWR analysis outputs

Example:
    >>> # Automatic fallback: tries Redis, then Memcached
    >>> cache = FallbackCache(
    ...     redis_url="redis://localhost:6379",
    ...     memcached_servers=["localhost:11211"]
    ... )
    >>> await cache.connect()
    >>> await cache.set("key", {"data": "value"})
"""

from typing import Optional, Any, Dict, List
from abc import ABC, abstractmethod
import json
import hashlib
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

# Optional Redis import
try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available")

# Optional Memcached import
try:
    import aiomcache
    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False
    logger.warning("Memcached (aiomcache) not available")


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def connect(self):
        """Connect to cache backend."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from cache backend."""
        pass

    @abstractmethod
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set cached value with TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cached value."""
        pass

    @abstractmethod
    async def clear_all(self) -> bool:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if cache backend is available."""
        pass


class RedisCache(CacheBackend):
    """
    Redis-based cache for SIE-X.
    
    Features:
    - Automatic key generation from content
    - TTL support
    - Async operations
    - Graceful fallback if Redis unavailable
    
    Example:
        >>> cache = RedisCache("redis://localhost:6379")
        >>> await cache.set("key", {"data": "value"}, ttl=3600)
        >>> result = await cache.get("key")
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 3600,
        key_prefix: str = "siex:"
    ):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds (1 hour)
            key_prefix: Prefix for all keys
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.client: Optional[aioredis.Redis] = None
        self.enabled = REDIS_AVAILABLE
        
        if not REDIS_AVAILABLE:
            logging.warning("Redis client not available, caching disabled")

    @property
    def is_available(self) -> bool:
        """Check if Redis is available and connected."""
        return self.enabled and self.client is not None

    async def connect(self):
        """Connect to Redis."""
        if not self.enabled:
            return
        
        try:
            self.client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.client.ping()
            logging.info("Redis cache connected")
        except Exception as e:
            logging.error(f"Redis connection failed: {e}")
            self.enabled = False
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
    
    def _generate_key(self, text: str, **params) -> str:
        """Generate cache key from text and parameters."""
        # Create hash of text + params
        content = text + json.dumps(params, sort_keys=True)
        hash_key = hashlib.md5(content.encode()).hexdigest()
        return f"{self.key_prefix}{hash_key}"
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value."""
        if not self.enabled or not self.client:
            return None
        
        try:
            data = await self.client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logging.error(f"Cache get error: {e}")
        
        return None
    
    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set cached value with TTL."""
        if not self.enabled or not self.client:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            await self.client.setex(
                key,
                ttl,
                json.dumps(value)
            )
            return True
        except Exception as e:
            logging.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete cached value."""
        if not self.enabled or not self.client:
            return False
        
        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logging.error(f"Cache delete error: {e}")
            return False
    
    async def clear_all(self) -> bool:
        """Clear all SIE-X cache entries."""
        if not self.enabled or not self.client:
            return False
        
        try:
            keys = await self.client.keys(f"{self.key_prefix}*")
            if keys:
                await self.client.delete(*keys)
            return True
        except Exception as e:
            logging.error(f"Cache clear error: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled or not self.client:
            return {"enabled": False}
        
        try:
            info = await self.client.info()
            keys_count = await self.client.dbsize()
            
            return {
                "enabled": True,
                "connected": True,
                "keys_count": keys_count,
                "memory_used": info.get("used_memory_human", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            logging.error(f"Cache stats error: {e}")
            return {"enabled": True, "connected": False, "error": str(e)}


class MemcachedCache(CacheBackend):
    """
    Memcached-based cache for SIE-X (fallback option).

    Features:
    - Lighter weight than Redis
    - Widely available in hosting environments
    - Good fallback for basic caching needs
    - Async operations via aiomcache

    Example:
        >>> cache = MemcachedCache(servers=["localhost:11211"])
        >>> await cache.connect()
        >>> await cache.set("key", {"data": "value"}, ttl=3600)
    """

    def __init__(
        self,
        servers: List[str] = None,
        default_ttl: int = 3600,
        key_prefix: str = "siex:"
    ):
        """
        Initialize Memcached cache.

        Args:
            servers: List of memcached servers (host:port format)
            default_ttl: Default TTL in seconds (1 hour)
            key_prefix: Prefix for all keys
        """
        self.servers = servers or ["localhost:11211"]
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.client: Optional[aiomcache.Client] = None
        self.enabled = MEMCACHED_AVAILABLE

        if not MEMCACHED_AVAILABLE:
            logger.warning("Memcached client not available, caching disabled")

    @property
    def is_available(self) -> bool:
        """Check if Memcached is available and connected."""
        return self.enabled and self.client is not None

    async def connect(self):
        """Connect to Memcached."""
        if not self.enabled:
            return

        try:
            # Parse first server
            host, port = self.servers[0].split(":")
            self.client = aiomcache.Client(host, int(port))

            # Test connection
            await self.client.get(b"test")
            logger.info(f"Memcached cache connected to {self.servers[0]}")
        except Exception as e:
            logger.error(f"Memcached connection failed: {e}")
            self.enabled = False

    async def disconnect(self):
        """Disconnect from Memcached."""
        if self.client:
            self.client.close()

    def _generate_key(self, text: str, **params) -> bytes:
        """Generate cache key from text and parameters."""
        content = text + json.dumps(params, sort_keys=True)
        hash_key = hashlib.md5(content.encode()).hexdigest()
        return f"{self.key_prefix}{hash_key}".encode()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value."""
        if not self.enabled or not self.client:
            return None

        try:
            if isinstance(key, str):
                key = key.encode()
            data = await self.client.get(key)
            if data:
                return json.loads(data.decode())
        except Exception as e:
            logger.error(f"Memcached get error: {e}")

        return None

    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set cached value with TTL."""
        if not self.enabled or not self.client:
            return False

        try:
            if isinstance(key, str):
                key = key.encode()
            ttl = ttl or self.default_ttl
            data = json.dumps(value).encode()
            await self.client.set(key, data, exptime=ttl)
            return True
        except Exception as e:
            logger.error(f"Memcached set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete cached value."""
        if not self.enabled or not self.client:
            return False

        try:
            if isinstance(key, str):
                key = key.encode()
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Memcached delete error: {e}")
            return False

    async def clear_all(self) -> bool:
        """Clear all cache entries."""
        if not self.enabled or not self.client:
            return False

        try:
            await self.client.flush_all()
            return True
        except Exception as e:
            logger.error(f"Memcached clear error: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled or not self.client:
            return {"enabled": False, "backend": "memcached"}

        try:
            stats = await self.client.stats()
            return {
                "enabled": True,
                "backend": "memcached",
                "connected": True,
                "stats": stats
            }
        except Exception as e:
            logger.error(f"Memcached stats error: {e}")
            return {"enabled": True, "backend": "memcached", "connected": False, "error": str(e)}


class FallbackCache(CacheBackend):
    """
    Multi-backend cache with automatic fallback.

    Strategy:
    1. Try primary backend (Redis)
    2. On failure, try secondary backend (Memcached)
    3. If all fail, operations return None/False gracefully

    This provides maximum deployment flexibility - works with Redis,
    Memcached, or both. Automatically degrades gracefully.

    Example:
        >>> cache = FallbackCache(
        ...     redis_url="redis://localhost:6379",
        ...     memcached_servers=["localhost:11211"]
        ... )
        >>> await cache.connect()
        >>> # Will use Redis if available, Memcached as fallback
        >>> await cache.set("key", {"data": "value"})
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        memcached_servers: List[str] = None,
        default_ttl: int = 3600,
        key_prefix: str = "siex:"
    ):
        """
        Initialize fallback cache with multiple backends.

        Args:
            redis_url: Redis connection URL (primary)
            memcached_servers: Memcached servers (fallback)
            default_ttl: Default TTL in seconds
            key_prefix: Prefix for all keys
        """
        self.backends: List[CacheBackend] = []
        self.active_backend: Optional[CacheBackend] = None
        self.key_prefix = key_prefix

        # Try to initialize Redis (primary)
        if REDIS_AVAILABLE:
            self.backends.append(RedisCache(redis_url, default_ttl, key_prefix))

        # Try to initialize Memcached (fallback)
        if MEMCACHED_AVAILABLE:
            servers = memcached_servers or ["localhost:11211"]
            self.backends.append(MemcachedCache(servers, default_ttl, key_prefix))

        if not self.backends:
            logger.warning("No cache backends available")

    @property
    def is_available(self) -> bool:
        """Check if any backend is available."""
        return self.active_backend is not None and self.active_backend.is_available

    async def connect(self):
        """Connect to first available backend."""
        for backend in self.backends:
            await backend.connect()
            if backend.is_available:
                self.active_backend = backend
                backend_name = backend.__class__.__name__
                logger.info(f"FallbackCache using: {backend_name}")
                return

        logger.warning("FallbackCache: No backends connected successfully")

    async def disconnect(self):
        """Disconnect all backends."""
        for backend in self.backends:
            await backend.disconnect()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get from active backend, try fallback on failure."""
        if not self.active_backend:
            return None

        try:
            result = await self.active_backend.get(key)
            return result
        except Exception as e:
            logger.error(f"Primary cache get failed: {e}")
            # Try fallback backends
            for backend in self.backends:
                if backend != self.active_backend and backend.is_available:
                    try:
                        return await backend.get(key)
                    except Exception:
                        continue
            return None

    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set in active backend, try fallback on failure."""
        if not self.active_backend:
            return False

        try:
            return await self.active_backend.set(key, value, ttl)
        except Exception as e:
            logger.error(f"Primary cache set failed: {e}")
            # Try fallback backends
            for backend in self.backends:
                if backend != self.active_backend and backend.is_available:
                    try:
                        return await backend.set(key, value, ttl)
                    except Exception:
                        continue
            return False

    async def delete(self, key: str) -> bool:
        """Delete from all backends."""
        success = False
        for backend in self.backends:
            if backend.is_available:
                try:
                    if await backend.delete(key):
                        success = True
                except Exception:
                    continue
        return success

    async def clear_all(self) -> bool:
        """Clear all backends."""
        success = False
        for backend in self.backends:
            if backend.is_available:
                try:
                    if await backend.clear_all():
                        success = True
                except Exception:
                    continue
        return success

    async def get_stats(self) -> Dict[str, Any]:
        """Get stats from all backends."""
        stats = {"fallback_cache": True, "backends": []}

        for backend in self.backends:
            try:
                backend_stats = await backend.get_stats()
                backend_stats["backend_type"] = backend.__class__.__name__
                backend_stats["is_active"] = (backend == self.active_backend)
                stats["backends"].append(backend_stats)
            except Exception as e:
                stats["backends"].append({
                    "backend_type": backend.__class__.__name__,
                    "error": str(e)
                })

        return stats
