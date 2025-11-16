"""
Metrics collection for SIE-X monitoring and observability.

Provides:
- Request/response metrics
- Performance timing
- Error tracking
- Cache hit/miss rates
- SEO analysis metrics
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import time
import statistics


@dataclass
class Metric:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collect and aggregate metrics for SIE-X.
    
    Features:
    - Counter metrics (requests, errors)
    - Histogram metrics (latency, processing time)
    - Gauge metrics (active connections, cache size)
    - Time-series storage (last 1000 points)
    
    Example:
        >>> metrics = MetricsCollector()
        >>> metrics.increment("requests_total")
        >>> with metrics.timer("extraction_time"):
        ...     # Do extraction
        ...     pass
        >>> stats = metrics.get_stats()
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Max number of historical points to keep
        """
        self.max_history = max_history
        
        # Counters
        self.counters: Dict[str, int] = defaultdict(int)
        
        # Histograms (time series)
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        
        # Gauges (current values)
        self.gauges: Dict[str, float] = {}
        
        # Errors
        self.errors: deque = deque(maxlen=100)
        
        # Start time
        self.start_time = datetime.now()
    
    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        key = self._make_key(name, tags)
        self.counters[key] += value
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value (e.g., latency)."""
        key = self._make_key(name, tags)
        self.histograms[key].append(value)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge value (e.g., active connections)."""
        key = self._make_key(name, tags)
        self.gauges[key] = value
    
    def record_error(self, error_type: str, message: str, details: Optional[Dict] = None):
        """Record an error."""
        self.errors.append({
            "type": error_type,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })
        self.increment("errors_total", tags={"type": error_type})
    
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        return _TimerContext(self, name, tags)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all metrics statistics."""
        return {
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": self._calculate_histogram_stats(),
            "recent_errors": list(self.errors)[-10:],  # Last 10 errors
            "summary": self._generate_summary()
        }
    
    def _calculate_histogram_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for all histograms."""
        stats = {}
        
        for name, values in self.histograms.items():
            if not values:
                continue
            
            values_list = list(values)
            stats[name] = {
                "count": len(values_list),
                "min": min(values_list),
                "max": max(values_list),
                "mean": statistics.mean(values_list),
                "median": statistics.median(values_list),
                "p95": self._percentile(values_list, 0.95),
                "p99": self._percentile(values_list, 0.99)
            }
        
        return stats
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate high-level summary."""
        total_requests = self.counters.get("requests_total", 0)
        total_errors = self.counters.get("errors_total", 0)
        
        success_rate = (
            (total_requests - total_errors) / total_requests 
            if total_requests > 0 
            else 0.0
        )
        
        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "success_rate": success_rate,
            "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
        }
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create metric key from name and tags."""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"
    
    def reset(self):
        """Reset all metrics."""
        self.counters.clear()
        self.histograms.clear()
        self.gauges.clear()
        self.errors.clear()
        self.start_time = datetime.now()


class _TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]]):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.collector.record(self.name, duration, self.tags)
        
        # Also increment counter
        self.collector.increment(f"{self.name}_count", tags=self.tags)
        
        # Record error if exception occurred
        if exc_type is not None:
            self.collector.record_error(
                error_type=exc_type.__name__,
                message=str(exc_val),
                details={"operation": self.name, "tags": self.tags}
            )
        
        return False  # Don't suppress exceptions


# Global metrics collector instance
_global_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _global_metrics
