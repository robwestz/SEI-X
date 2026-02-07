"""
Metrics collection for SIE-X monitoring and observability.
Uses prometheus_client for standard metric exposition.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

# --- Prometheus Metrics Definitions ---

REQUESTS_TOTAL = Counter(
    'sie_x_requests_total',
    'Total number of requests processed',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'sie_x_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)

ERRORS_TOTAL = Counter(
    'sie_x_errors_total',
    'Total number of errors',
    ['type']
)

ACTIVE_REQUESTS = Gauge(
    'sie_x_active_requests',
    'Number of requests currently being processed'
)

KEYWORDS_EXTRACTED = Counter(
    'sie_x_keywords_extracted_total',
    'Total number of keywords extracted'
)

MODEL_INFO = Gauge(
    'sie_x_model_info',
    'Information about loaded models',
    ['model_name', 'spacy_model']
)

def get_metrics_app():
    """Returns an ASGI app to serve metrics."""
    return make_asgi_app()


# --- Legacy/Internal Collector (Optional Wrapper) ---

@dataclass
class Metric:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    """
    Legacy wrapper for metrics.
    Now delegates to Prometheus metrics where applicable.
    """

    def __init__(self):
        self.start_time = datetime.now()
        self._timers: Dict[str, float] = {}

    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter."""
        # Map legacy names to Prometheus if needed, or just log
        pass

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        pass

    def start_timer(self, name: str):
        """Start a named timer."""
        self._timers[name] = time.time()

    def end_timer(self, name: str) -> float:
        """End a named timer and return duration in seconds."""
        if name in self._timers:
            duration = time.time() - self._timers[name]
            del self._timers[name]
            return duration
        return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get simple internal stats."""
        return {
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
        }

# Global metrics collector instance (Legacy)
_global_metrics = MetricsCollector()

def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _global_metrics