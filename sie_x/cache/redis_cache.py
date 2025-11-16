"""
Redis caching layer for SIE-X production deployment.

Provides distributed caching for:
- Keyword extraction results
- Semantic embeddings
- SEO analysis results
- BACOWR analysis outputs
"""

from typing import Optional, Any, Dict
import json
import hashlib
import logging
from datetime import timedelta

try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Redis not available, caching disabled")


class RedisCache:
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
