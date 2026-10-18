"""Tests for inv_store JSONL mtime cache.

Mutation notes (applied, confirmed red, then reverted):
- test_cold_read_populates_cache: removed ``_jsonl_cache_set`` call in ``_read_jsonl``
  → ``str(p) in _JSONL_CACHE`` fails (entry never stored). ✓
- test_warm_read_returns_cached: removed ``if cached is not None: return cached`` guard
  → ``_read_jsonl`` re-reads disk (3 rows) instead of returning cached (1 row);
  ``assert len(result) == 1`` fails. ✓
- test_mtime_change_triggers_cold_read: replaced entire ``_read_jsonl`` body with
  ``return []`` → ``assert len(result) == 2`` fails. ✓
- test_rewrite_invalidates_cache: removed ``_jsonl_cache_invalidate`` call from
  ``_rewrite_jsonl_preserving`` → stale entry remains; ``assert str(p) not in _JSONL_CACHE``
  fails. ✓  The ``updated["v"] == 99`` assertion then also proves the re-read picked
  up the rewritten file, not a stale cache copy.
- test_append_invalidates_cache: removed ``_jsonl_cache_invalidate`` call from
  ``_append_jsonl`` → stale entry remains; ``assert str(p) not in _JSONL_CACHE``
  fails. ✓
- test_missing_file_returns_empty / test_existing_file_returns_rows: removed the
  ``except OSError: return []`` path → existing-file test still passes, missing-file
  test raises OSError instead of returning []. ✓  (positive twin keeps the
  always-empty mutation from passing both.)
- test_cache_evicts_at_max: replaced ``if len(_JSONL_CACHE) >= _JSONL_CACHE_MAX``
  with ``if False`` → cache grows past cap; ``assert len(inv_store._JSONL_CACHE) == 64``
  fails (it equals 65). ✓
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import inv_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Cold-read populates cache
# ---------------------------------------------------------------------------

def test_cold_read_populates_cache(tmp_path, monkeypatch):
    """A cache miss reads the file and stores the result under the path key."""
    monkeypatch.setattr(inv_store, "_JSONL_CACHE", {})
    p = tmp_path / "test.jsonl"
    _write_jsonl(p, [{"id": "1"}, {"id": "2"}])

    result = inv_store._read_jsonl(p)

    assert len(result) == 2
    assert result[0]["id"] == "1"
    assert result[1]["id"] == "2"
    assert str(p) in inv_store._JSONL_CACHE


# ---------------------------------------------------------------------------
# Warm read returns cached copy (mtime unchanged)
# ---------------------------------------------------------------------------

def test_warm_read_returns_cached(tmp_path, monkeypatch):
    """A cache hit returns the stored rows without re-reading the file."""
    monkeypatch.setattr(inv_store, "_JSONL_CACHE", {})
    p = tmp_path / "test.jsonl"
    _write_jsonl(p, [{"id": "1"}])
    inv_store._read_jsonl(p)  # prime cache

    # Overwrite file content but restore the original mtime so the cache key matches.
    original_mtime = os.stat(p).st_mtime
    _write_jsonl(p, [{"id": "1"}, {"id": "2"}, {"id": "3"}])
    os.utime(p, (original_mtime, original_mtime))

    result = inv_store._read_jsonl(p)

    # Cache returned 1 row (the original snapshot), not 3 from the updated file.
    assert len(result) == 1
    assert result[0]["id"] == "1"


# ---------------------------------------------------------------------------
# mtime change triggers cold read
# ---------------------------------------------------------------------------

def test_mtime_change_triggers_cold_read(tmp_path, monkeypatch):
    """When the file's mtime changes the cache is bypassed and the file is re-read."""
    monkeypatch.setattr(inv_store, "_JSONL_CACHE", {})
    p = tmp_path / "test.jsonl"
    _write_jsonl(p, [{"id": "1"}])
    inv_store._read_jsonl(p)  # prime cache

    time.sleep(0.01)
    _write_jsonl(p, [{"id": "1"}, {"id": "2"}])  # new mtime

    result = inv_store._read_jsonl(p)

    assert len(result) == 2
    assert result[1]["id"] == "2"


# ---------------------------------------------------------------------------
# _rewrite_jsonl_preserving invalidates cache
# ---------------------------------------------------------------------------

def test_rewrite_invalidates_cache(tmp_path, monkeypatch):
    """Cache entry is removed synchronously when _rewrite_jsonl_preserving writes."""
    monkeypatch.setattr(inv_store, "_JSONL_CACHE", {})
    p = tmp_path / "test.jsonl"
    _write_jsonl(p, [{"id": "1", "v": 1}, {"id": "2", "v": 2}])
    inv_store._read_jsonl(p)  # prime cache

    # Replace the first row (return None to keep a row unchanged, dict to replace it).
    inv_store._rewrite_jsonl_preserving(p, lambda x: {**x, "v": 99} if x["id"] == "1" else None)

    assert str(p) not in inv_store._JSONL_CACHE

    # Positive check: re-reading reflects the rewritten content from disk.
    result = inv_store._read_jsonl(p)
    assert len(result) == 2
    updated = next(r for r in result if r["id"] == "1")
    assert updated["v"] == 99


# ---------------------------------------------------------------------------
# _append_jsonl invalidates cache
# ---------------------------------------------------------------------------

def test_append_invalidates_cache(tmp_path, monkeypatch):
    """Cache entry is removed when _append_jsonl writes a new row."""
    monkeypatch.setattr(inv_store, "_JSONL_CACHE", {})
    p = tmp_path / "test.jsonl"
    _write_jsonl(p, [{"id": "1"}])
    inv_store._read_jsonl(p)  # prime cache

    inv_store._append_jsonl(p, {"id": "2"})

    assert str(p) not in inv_store._JSONL_CACHE

    # Positive check: re-reading gives both rows.
    result = inv_store._read_jsonl(p)
    assert len(result) == 2
    assert result[1]["id"] == "2"


# ---------------------------------------------------------------------------
# Missing file returns [] (no OSError leaks)
# ---------------------------------------------------------------------------

def test_missing_file_returns_empty(tmp_path, monkeypatch):
    """A path that does not exist returns an empty list without raising."""
    monkeypatch.setattr(inv_store, "_JSONL_CACHE", {})
    result = inv_store._read_jsonl(tmp_path / "nonexistent.jsonl")
    assert result == []


def test_existing_file_returns_rows(tmp_path, monkeypatch):
    """Positive twin: a file that *does* exist returns its rows."""
    monkeypatch.setattr(inv_store, "_JSONL_CACHE", {})
    p = tmp_path / "exists.jsonl"
    _write_jsonl(p, [{"id": "x"}])
    result = inv_store._read_jsonl(p)
    assert len(result) == 1
    assert result[0]["id"] == "x"


# ---------------------------------------------------------------------------
# LRU eviction at _JSONL_CACHE_MAX
# ---------------------------------------------------------------------------

def test_cache_evicts_at_max(tmp_path, monkeypatch):
    """Cache never grows beyond _JSONL_CACHE_MAX entries; oldest is evicted."""
    fresh_cache: dict = {}
    monkeypatch.setattr(inv_store, "_JSONL_CACHE", fresh_cache)
    cap = inv_store._JSONL_CACHE_MAX  # 64

    # Fill the cache to capacity using distinct files.
    first_path = None
    for i in range(cap):
        p = tmp_path / f"f{i}.jsonl"
        _write_jsonl(p, [{"id": str(i)}])
        inv_store._read_jsonl(p)
        if first_path is None:
            first_path = p

    assert len(fresh_cache) == cap
    assert str(first_path) in fresh_cache  # oldest still present

    # One more file should evict the oldest entry.
    extra = tmp_path / "extra.jsonl"
    _write_jsonl(extra, [{"id": "extra"}])
    inv_store._read_jsonl(extra)

    assert len(fresh_cache) == cap
    assert str(first_path) not in fresh_cache  # evicted
    assert str(extra) in fresh_cache           # newest present


# ---------------------------------------------------------------------------
# Returned list is an independent copy (mutations don't corrupt the cache)
# ---------------------------------------------------------------------------

def test_read_returns_independent_copy(tmp_path, monkeypatch):
    """Mutating the returned list must not alter the cached snapshot."""
    monkeypatch.setattr(inv_store, "_JSONL_CACHE", {})
    p = tmp_path / "test.jsonl"
    _write_jsonl(p, [{"id": "1"}])
    inv_store._read_jsonl(p)  # prime cache

    first = inv_store._read_jsonl(p)
    first.append({"id": "injected"})  # mutate the returned copy

    second = inv_store._read_jsonl(p)
    assert len(second) == 1  # cache is unaffected
    assert second[0]["id"] == "1"
