"""
In-memory cache for lightweight usage and tests.
"""

from collections import OrderedDict
from typing import Any, Optional


class CacheManager:
    """Simple LRU cache with a max size limit."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._store: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
