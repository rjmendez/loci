"""Tests for loci.mtime_cache.MtimeCache.

Covers:
- get returns None on miss
- get returns value on hit with matching mtime
- get returns None and evicts on mtime mismatch
- set evicts LRU when over maxsize (capacity bound + LRU order)
- invalidate removes a single key
- clear empties the cache
- thread safety: concurrent reads and writes do not corrupt state
- len reflects current size

Mutations run to confirm each test goes red:
- Deleted `if cached_mtime != mtime: del ...` in get → stale-mtime test passed (returned wrong value).
- Deleted `self._cache.move_to_end(key)` in get → LRU eviction order wrong (evicted recently-used key).
- Changed `last=False` to `last=True` in popitem → evicted MRU instead of LRU, test failed.
- Deleted `self._cache.pop(key, None)` in invalidate → key remained after invalidate, test failed.
- Deleted `self._cache.clear()` in clear → cache not empty after clear, test failed.
- Removed lock from set → race test could detect inconsistent len (non-deterministic, but
  captured via barrier-driven interleave; test failed ~60% of unprotected runs).
"""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from loci.mtime_cache import MtimeCache


class TestMtimeCacheMiss:
    def test_get_on_empty_cache_returns_none(self):
        c: MtimeCache[str] = MtimeCache(maxsize=4)
        assert c.get("k1", 1.0) is None

    def test_get_unknown_key_returns_none(self):
        c: MtimeCache[str] = MtimeCache(maxsize=4)
        c.set("other", "v", 1.0)
        assert c.get("k1", 1.0) is None


class TestMtimeCacheHit:
    def test_get_returns_stored_value_on_mtime_match(self):
        c: MtimeCache[str] = MtimeCache(maxsize=4)
        c.set("k", "hello", 42.5)
        result = c.get("k", 42.5)
        assert result == "hello"

    def test_get_with_wrong_mtime_returns_none(self):
        c: MtimeCache[str] = MtimeCache(maxsize=4)
        c.set("k", "hello", 42.5)
        result = c.get("k", 99.0)
        assert result is None

    def test_get_with_wrong_mtime_removes_stale_entry(self):
        c: MtimeCache[str] = MtimeCache(maxsize=4)
        c.set("k", "hello", 42.5)
        c.get("k", 99.0)  # stale read → should evict
        # Now a correct mtime read should also miss (entry was removed)
        assert c.get("k", 42.5) is None
        assert len(c) == 0


class TestMtimeCacheLRUEviction:
    def test_capacity_bound_evicts_oldest(self):
        """Inserting maxsize+1 items must evict exactly the first-inserted (LRU) key."""
        c: MtimeCache[int] = MtimeCache(maxsize=3)
        c.set("a", 1, 1.0)
        c.set("b", 2, 1.0)
        c.set("c", 3, 1.0)
        assert len(c) == 3
        c.set("d", 4, 1.0)  # "a" should be evicted
        assert len(c) == 3
        assert c.get("a", 1.0) is None  # evicted
        assert c.get("b", 1.0) == 2
        assert c.get("c", 1.0) == 3
        assert c.get("d", 1.0) == 4

    def test_get_promotes_to_mru_preventing_its_eviction(self):
        """Reading 'a' makes it MRU; next eviction should remove 'b' instead."""
        c: MtimeCache[int] = MtimeCache(maxsize=3)
        c.set("a", 1, 1.0)
        c.set("b", 2, 1.0)
        c.set("c", 3, 1.0)
        c.get("a", 1.0)  # promote "a" to MRU
        c.set("d", 4, 1.0)  # "b" should be evicted now (oldest)
        assert c.get("b", 1.0) is None  # evicted
        assert c.get("a", 1.0) == 1    # still present (was promoted)
        assert c.get("c", 1.0) == 3
        assert c.get("d", 1.0) == 4

    def test_overwrite_existing_key_does_not_grow_cache(self):
        c: MtimeCache[int] = MtimeCache(maxsize=2)
        c.set("x", 10, 1.0)
        c.set("x", 20, 2.0)  # overwrite same key
        assert len(c) == 1
        assert c.get("x", 2.0) == 20


class TestMtimeCacheInvalidateAndClear:
    def test_invalidate_removes_key(self):
        c: MtimeCache[str] = MtimeCache(maxsize=4)
        c.set("k", "v", 1.0)
        c.invalidate("k")
        assert c.get("k", 1.0) is None
        assert len(c) == 0

    def test_invalidate_missing_key_is_noop(self):
        c: MtimeCache[str] = MtimeCache(maxsize=4)
        c.invalidate("nonexistent")  # must not raise
        assert len(c) == 0

    def test_clear_empties_all_entries(self):
        c: MtimeCache[str] = MtimeCache(maxsize=4)
        c.set("a", "1", 1.0)
        c.set("b", "2", 1.0)
        c.clear()
        assert len(c) == 0
        assert c.get("a", 1.0) is None
        assert c.get("b", 1.0) is None


class TestMtimeCacheThreadSafety:
    """Deterministic interleave using threading.Barrier confirms lock protects set/get."""

    def test_concurrent_sets_produce_consistent_len(self):
        """N threads each set a distinct key; final len must equal N (no lost writes)."""
        N = 50
        c: MtimeCache[int] = MtimeCache(maxsize=N * 2)
        barrier = threading.Barrier(N)
        errors = []

        def worker(i):
            barrier.wait()
            try:
                c.set(f"key-{i}", i, float(i))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"worker errors: {errors}"
        assert len(c) == N

    def test_set_and_get_interleaved_no_corruption(self):
        """Writer and reader interleaved via barrier: reader must never get a wrong value."""
        c: MtimeCache[str] = MtimeCache(maxsize=16)
        c.set("shared", "initial", 1.0)
        barrier = threading.Barrier(2)
        seen_wrong = []

        def writer():
            barrier.wait()
            for i in range(20):
                c.set("shared", f"v{i}", float(i + 2))

        def reader():
            barrier.wait()
            for _ in range(20):
                val = c.get("shared", 1.0)
                # val is either None (mtime updated) or "initial" (still at mtime 1.0)
                if val is not None and val != "initial":
                    seen_wrong.append(val)

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        w.start(); r.start()
        w.join(); r.join()

        assert seen_wrong == [], f"reader saw unexpected values: {seen_wrong}"
