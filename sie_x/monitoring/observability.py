"""
Production observability with structured logging and metrics.
"""

import time
import json
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import asynccontextmanager
import structlog
from prometheus_client import Counter, Histogram, Gauge, Info
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Prometheus metrics
extraction_counter = Counter(
    'sie_x_extractions_total',
    'Total number of extractions',
    ['mode', 'status']
)

extraction_duration = Histogram(
    'sie_x_extraction_duration_seconds',
    'Duration of extraction operations',
    ['mode', 'document_size_category']
)

active_operations = Gauge(
    'sie_x_active_operations',
    'Number of active operations',
    ['operation_type']
)

model_info = Info(
    'sie_x_model',
    'Information about loaded models'
)

cache_metrics = {
    'hits': Counter('sie_x_cache_hits_total', 'Cache hits'),
    'misses': Counter('sie_x_cache_misses_total', 'Cache misses'),
    'size': Gauge('sie_x_cache_size_bytes', 'Current cache size in bytes')
}

# OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add OTLP exporter for Jaeger/Tempo
otlp_exporter = OTLPSpanExporter(
    endpoint="localhost:4317",
    insecure=True
)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)


@dataclass
class OperationContext:
    """Context for tracking operations."""
    operation_id: str
    operation_type: str
    start_time: float
    metadata: Dict[str, Any]

    def duration(self) -> float:
        return time.time() - self.start_time


class ObservabilityManager:
    """Central observability management."""

    def __init__(self):
        self.logger = structlog.get_logger()
        self._contexts: Dict[str, OperationContext] = {}

    @asynccontextmanager
    async def track_operation(
            self,
            operation_type: str,
            operation_id: str,
            **metadata
    ):
        """Track an operation with full observability."""
        # Start tracking
        context = OperationContext(
            operation_id=operation_id,
            operation_type=operation_type,
            start_time=time.time(),
            metadata=metadata
        )
        self._contexts[operation_id] = context

        # Update metrics
        active_operations.labels(operation_type=operation_type).inc()

        # Start trace span
        with tracer.start_as_current_span(
                operation_type,
                attributes=metadata
        ) as span:
            try:
                # Log operation start
                self.logger.info(
                    "operation_started",
                    operation_id=operation_id,
                    operation_type=operation_type,
                    **metadata
                )

                yield context

                # Success metrics
                extraction_counter.labels(
                    mode=metadata.get('mode', 'unknown'),
                    status='success'
                ).inc()

                span.set_status(trace.Status(trace.StatusCode.OK))

            except Exception as e:
                # Failure metrics
                extraction_counter.labels(
                    mode=metadata.get('mode', 'unknown'),
                    status='error'
                ).inc()

                # Log error with full context
                self.logger.error(
                    "operation_failed",
                    operation_id=operation_id,
                    operation_type=operation_type,
                    error=str(e),
                    error_type=type(e).__name__,
                    duration=context.duration(),
                    **metadata,
                    exc_info=True
                )

                span.record_exception(e)
                span.set_status(
                    trace.Status(
                        trace.StatusCode.ERROR,
                        str(e)
                    )
                )

                raise

            finally:
                # Clean up
                active_operations.labels(operation_type=operation_type).dec()

                # Record duration
                duration = context.duration()
                doc_size = metadata.get('document_size', 0)
                size_category = self._categorize_size(doc_size)

                extraction_duration.labels(
                    mode=metadata.get('mode', 'unknown'),
                    document_size_category=size_category
                ).observe(duration)

                # Log completion
                self.logger.info(
                    "operation_completed",
                    operation_id=operation_id,
                    operation_type=operation_type,
                    duration=duration,
                    **metadata
                )

                # Clean up context
                self._contexts.pop(operation_id, None)

    def _categorize_size(self, size: int) -> str:
        """Categorize document size for metrics."""
        if size < 1000:
            return "small"
        elif size < 10000:
            return "medium"
        elif size < 100000:
            return "large"
        else:
            return "xlarge"

    def log_cache_event(self, event_type: str, key: str, size: Optional[int] = None):
        """Log cache events with metrics."""
        if event_type == 'hit':
            cache_metrics['hits'].inc()
            self.logger.debug("cache_hit", key=key)
        elif event_type == 'miss':
            cache_metrics['misses'].inc()
            self.logger.debug("cache_miss", key=key)

        if size is not None:
            cache_metrics['size'].set(size)

    def record_model_info(self, model_name: str, version: str, **info):
        """Record model information."""
        model_info.info({
            'model_name': model_name,
            'version': version,
            **info
        })


class PerformanceMonitor:
    """Monitor performance metrics."""

    def __init__(self):
        self.metrics = {}

    async def measure_latency(self, operation: Callable, *args, **kwargs):
        """Measure operation latency."""
        start = time.perf_counter()
        try:
            result = await operation(*args, **kwargs)
            latency = time.perf_counter() - start

            self.metrics[operation.__name__] = {
                'latency': latency,
                'timestamp': datetime.utcnow().isoformat(),
                'success': True
            }

            return result

        except Exception as e:
            latency = time.perf_counter() - start

            self.metrics[operation.__name__] = {
                'latency': latency,
                'timestamp': datetime.utcnow().isoformat(),
                'success': False,
                'error': str(e)
            }

            raise