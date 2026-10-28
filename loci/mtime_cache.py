"""Shared mtime-keyed LRU cache for file-backed data."""

import threading
from collections import OrderedDict
from typing import Generic, Optional, TypeVar

V = TypeVar("V")


class MtimeCache(Generic[V]):
    """
    Thread-safe LRU cache keyed by (str_key, file_mtime).

    On get: returns cached value if mtime matches, else None.
    On set: stores value with mtime; evicts LRU if over maxsize.
    """

    def __init__(self, maxsize: int = 128):
        self._cache: OrderedDict[str, tuple[V, float]] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str, mtime: float) -> Optional[V]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            value, cached_mtime = entry
            if cached_mtime != mtime:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: V, mtime: float) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, mtime)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)  # evict LRU (oldest)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
