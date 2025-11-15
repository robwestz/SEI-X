"""
Production-grade error handling and retry mechanisms.
"""

import asyncio
import functools
import logging
from typing import TypeVar, Callable, Any, Optional, Type, Tuple
from dataclasses import dataclass
from enum import Enum
import backoff
from circuitbreaker import circuit

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels for intelligent handling."""
    TRANSIENT = "transient"  # Retry immediately
    RECOVERABLE = "recoverable"  # Retry with backoff
    DEGRADED = "degraded"  # Fallback to simpler method
    CRITICAL = "critical"  # Fail fast


@dataclass
class ErrorContext:
    """Context for error handling decisions."""
    error_type: Type[Exception]
    severity: ErrorSeverity
    max_retries: int = 3
    fallback: Optional[Callable] = None
    alert_threshold: int = 2


class IntelligentRetry:
    """Adaptive retry mechanism with circuit breaker."""

    # Error classification map
    ERROR_MAP = {
        ConnectionError: ErrorSeverity.TRANSIENT,
        TimeoutError: ErrorSeverity.RECOVERABLE,
        MemoryError: ErrorSeverity.DEGRADED,
        ValueError: ErrorSeverity.CRITICAL,
        RuntimeError: ErrorSeverity.RECOVERABLE,
    }

    @staticmethod
    def classify_error(e: Exception) -> ErrorSeverity:
        """Classify error severity for intelligent handling."""
        for error_type, severity in IntelligentRetry.ERROR_MAP.items():
            if isinstance(e, error_type):
                return severity
        return ErrorSeverity.RECOVERABLE

    @staticmethod
    def with_fallback(
            primary_func: Callable[..., T],
            fallback_func: Callable[..., T],
            condition: Callable[[Exception], bool] = lambda e: True
    ) -> Callable[..., T]:
        """Decorator for fallback on failure."""

        @functools.wraps(primary_func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await primary_func(*args, **kwargs)
            except Exception as e:
                if condition(e):
                    logger.warning(f"Primary failed, using fallback: {e}")
                    return await fallback_func(*args, **kwargs)
                raise

        return wrapper


def resilient_operation(
        max_tries: int = 3,
        backoff_factor: float = 2.0,
        fallback: Optional[Callable] = None,
        circuit_failure_threshold: int = 5,
        circuit_recovery_timeout: int = 60
):
    """
    Decorator for resilient operations with retry, backoff, and circuit breaker.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Apply circuit breaker
        breaker_func = circuit(
            failure_threshold=circuit_failure_threshold,
            recovery_timeout=circuit_recovery_timeout,
            expected_exception=Exception
        )(func)

        # Apply exponential backoff
        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=max_tries,
            max_time=300,  # 5 minutes max
            factor=backoff_factor,
            on_backoff=lambda details: logger.warning(
                f"Retry {details['tries']}/{max_tries} for {func.__name__}"
            )
        )
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_error = None

            try:
                # Try the main operation
                return await breaker_func(*args, **kwargs)

            except Exception as e:
                last_error = e
                severity = IntelligentRetry.classify_error(e)

                # Handle based on severity
                if severity == ErrorSeverity.CRITICAL:
                    logger.error(f"Critical error in {func.__name__}: {e}")
                    raise

                elif severity == ErrorSeverity.DEGRADED and fallback:
                    logger.warning(f"Degraded mode, using fallback: {e}")
                    return await fallback(*args, **kwargs)

                # Let backoff handle other cases
                raise

            finally:
                # Log metrics
                if last_error:
                    logger.info(f"Operation {func.__name__} recovered from {type(last_error).__name__}")

        return wrapper

    return decorator


class ResourceManager:
    """Manage resources with automatic cleanup and limits."""

    def __init__(
            self,
            max_memory_mb: int = 4096,
            max_concurrent: int = 100,
            cleanup_interval: int = 300
    ):
        self.max_memory_mb = max_memory_mb
        self.max_concurrent = max_concurrent
        self.cleanup_interval = cleanup_interval
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._resources = {}

    async def acquire(self, resource_id: str, size_mb: float = 0):
        """Acquire resource with memory tracking."""
        async with self._semaphore:
            current_usage = sum(r.get('size', 0) for r in self._resources.values())

            if current_usage + size_mb > self.max_memory_mb:
                # Trigger cleanup
                await self._cleanup_old_resources()

                # Check again
                current_usage = sum(r.get('size', 0) for r in self._resources.values())
                if current_usage + size_mb > self.max_memory_mb:
                    raise MemoryError(f"Resource limit exceeded: {current_usage + size_mb}MB > {self.max_memory_mb}MB")

            self._resources[resource_id] = {
                'size': size_mb,
                'acquired_at': asyncio.get_event_loop().time()
            }

    async def release(self, resource_id: str):
        """Release resource."""
        self._resources.pop(resource_id, None)

    async def _cleanup_old_resources(self):
        """Clean up resources older than cleanup_interval."""
        current_time = asyncio.get_event_loop().time()
        to_remove = [
            rid for rid, info in self._resources.items()
            if current_time - info['acquired_at'] > self.cleanup_interval
        ]

        for rid in to_remove:
            logger.info(f"Cleaning up old resource: {rid}")
            self._resources.pop(rid, None)