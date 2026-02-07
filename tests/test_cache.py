"""Tests for in-memory CacheManager."""

import pytest
from unittest.mock import patch

from sie_x.cache.manager import CacheManager


class TestCacheBasic:

    def test_set_and_get(self):
        cache = CacheManager(max_size=10)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_miss_returns_none(self):
        cache = CacheManager(max_size=10)
        assert cache.get("nonexistent") is None

    def test_overwrite_existing_key(self):
        cache = CacheManager(max_size=10)
        cache.set("k", "old")
        cache.set("k", "new")
        assert cache.get("k") == "new"

    def test_stores_various_types(self):
        cache = CacheManager(max_size=10)
        cache.set("int", 42)
        cache.set("list", [1, 2, 3])
        cache.set("dict", {"a": 1})
        assert cache.get("int") == 42
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("dict") == {"a": 1}


class TestLRUEviction:

    def test_evicts_oldest_on_overflow(self):
        cache = CacheManager(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_access_refreshes_lru_order(self):
        cache = CacheManager(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # refresh "a", so "b" is oldest
        cache.set("c", 3)  # should evict "b"
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3


class TestClear:

    def test_clear_empties_cache(self):
        cache = CacheManager(max_size=10)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
