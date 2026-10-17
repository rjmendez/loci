"""Tests for mcp/graph_cache.py — deterministic graph tool result cache."""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from graph_cache import (
    GraphCache,
    GraphCacheEntry,
    make_cache_key,
)


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------


def test_make_cache_key_same_params_same_key():
    k1 = make_cache_key("code_graph_query", {"query": "foo", "depth": 2})
    k2 = make_cache_key("code_graph_query", {"query": "foo", "depth": 2})
    assert k1 == k2


def test_make_cache_key_different_params_different_key():
    k1 = make_cache_key("code_graph_query", {"query": "foo"})
    k2 = make_cache_key("code_graph_query", {"query": "bar"})
    assert k1 != k2


def test_make_cache_key_order_independent():
    k1 = make_cache_key("symbol_impact", {"a": 1, "b": 2})
    k2 = make_cache_key("symbol_impact", {"b": 2, "a": 1})
    assert k1 == k2


def test_make_cache_key_graph_revision_changes_key():
    k1 = make_cache_key("code_graph_query", {"q": "x"}, graph_revision="abc")
    k2 = make_cache_key("code_graph_query", {"q": "x"}, graph_revision="def")
    assert k1 != k2


def test_make_cache_key_length_16():
    k = make_cache_key("tool", {})
    assert len(k) == 16


# ---------------------------------------------------------------------------
# GraphCacheEntry
# ---------------------------------------------------------------------------


def test_entry_is_valid_when_fresh():
    entry = GraphCacheEntry(key="k", tool_name="t", params_hash="k", result={})
    assert entry.is_valid(ttl_minutes=30)


def test_entry_is_invalid_when_expired():
    old_ts = (datetime.now(timezone.utc) - timedelta(minutes=60)).isoformat()
    entry = GraphCacheEntry(key="k", tool_name="t", params_hash="k", result={}, cached_at=old_ts)
    assert not entry.is_valid(ttl_minutes=30)


def test_entry_roundtrip_dict():
    entry = GraphCacheEntry(key="abc", tool_name="symbol_impact", params_hash="abc", result={"nodes": [1, 2]})
    restored = GraphCacheEntry.from_dict(entry.to_dict())
    assert restored.key == entry.key
    assert restored.result == entry.result
    assert restored.tool_name == entry.tool_name


# ---------------------------------------------------------------------------
# GraphCache.put + get (roundtrip)
# ---------------------------------------------------------------------------


def test_put_and_get_roundtrip(tmp_path):
    cache = GraphCache(cache_dir=tmp_path, ttl_minutes=30)
    params = {"symbol": "loci.grounding", "depth": 3}
    result = {"edges": [{"from": "a", "to": "b"}], "count": 1}

    cache.put("symbol_impact", params, result)
    hit = cache.get("symbol_impact", params)

    assert hit is not None
    assert hit.result == result
    assert hit.tool_name == "symbol_impact"


def test_get_returns_none_on_miss(tmp_path):
    cache = GraphCache(cache_dir=tmp_path, ttl_minutes=30)
    result = cache.get("code_graph_query", {"q": "nonexistent"})
    assert result is None


def test_get_returns_none_after_ttl_expiry(tmp_path):
    cache = GraphCache(cache_dir=tmp_path, ttl_minutes=0.0001)
    params = {"q": "test"}
    cache.put("code_graph_query", params, {"data": 42})
    time.sleep(0.02)  # 20ms > 0.006s TTL
    result = cache.get("code_graph_query", params)
    assert result is None


def test_put_preserves_complex_result(tmp_path):
    cache = GraphCache(cache_dir=tmp_path, ttl_minutes=30)
    result = {"nodes": [{"id": 1, "label": "A"}, {"id": 2, "label": "B"}], "edges": [], "meta": None}
    cache.put("impact_report", {"target": "fn"}, result)
    hit = cache.get("impact_report", {"target": "fn"})
    assert hit.result == result


def test_different_tools_dont_collide(tmp_path):
    cache = GraphCache(cache_dir=tmp_path, ttl_minutes=30)
    params = {"q": "shared_params"}
    cache.put("tool_a", params, "result_a")
    cache.put("tool_b", params, "result_b")

    assert cache.get("tool_a", params).result == "result_a"
    assert cache.get("tool_b", params).result == "result_b"


# ---------------------------------------------------------------------------
# GraphCache.invalidate
# ---------------------------------------------------------------------------


def test_invalidate_single_tool_clears_only_that_file(tmp_path):
    cache = GraphCache(cache_dir=tmp_path, ttl_minutes=30)
    cache.put("code_graph_query", {"q": "x"}, "r1")
    cache.put("symbol_impact", {"s": "y"}, "r2")

    count = cache.invalidate("code_graph_query")

    assert count == 1
    assert cache.get("code_graph_query", {"q": "x"}) is None
    assert cache.get("symbol_impact", {"s": "y"}) is not None


def test_invalidate_none_clears_all_files(tmp_path):
    cache = GraphCache(cache_dir=tmp_path, ttl_minutes=30)
    cache.put("code_graph_query", {"q": "x"}, "r1")
    cache.put("symbol_impact", {"s": "y"}, "r2")
    cache.put("impact_report", {"t": "z"}, "r3")

    count = cache.invalidate()

    assert count == 3
    assert list(tmp_path.glob("*.jsonl")) == []


def test_invalidate_returns_zero_when_nothing_to_clear(tmp_path):
    cache = GraphCache(cache_dir=tmp_path, ttl_minutes=30)
    assert cache.invalidate("nonexistent_tool") == 0
    assert cache.invalidate() == 0


def test_invalidate_single_returns_correct_count(tmp_path):
    cache = GraphCache(cache_dir=tmp_path, ttl_minutes=30)
    cache.put("impact_report", {"t": "a"}, "ra")
    assert cache.invalidate("impact_report") == 1


# ---------------------------------------------------------------------------
# GraphCache.stats
# ---------------------------------------------------------------------------


def test_stats_reports_entries_per_tool(tmp_path):
    cache = GraphCache(cache_dir=tmp_path, ttl_minutes=30)
    cache.put("code_graph_query", {"q": "1"}, "r1")
    cache.put("code_graph_query", {"q": "2"}, "r2")
    cache.put("symbol_impact", {"s": "x"}, "rx")

    stats = cache.stats()

    assert "code_graph_query" in stats
    assert stats["code_graph_query"]["entries"] == 2
    assert "symbol_impact" in stats
    assert stats["symbol_impact"]["entries"] == 1


def test_stats_empty_cache(tmp_path):
    cache = GraphCache(cache_dir=tmp_path, ttl_minutes=30)
    assert cache.stats() == {}


def test_stats_includes_byte_size(tmp_path):
    cache = GraphCache(cache_dir=tmp_path, ttl_minutes=30)
    cache.put("subsystem_report", {"sub": "core"}, {"data": "x" * 100})
    stats = cache.stats()
    assert stats["subsystem_report"]["bytes"] > 0


# ---------------------------------------------------------------------------
# Full integration: put → get → invalidate → get returns None
# ---------------------------------------------------------------------------


def test_full_integration_put_get_invalidate_get(tmp_path):
    cache = GraphCache(cache_dir=tmp_path, ttl_minutes=30)
    params = {"symbol": "server.handle_request", "depth": 2}
    result = {"callers": ["a", "b"], "callees": ["c"]}

    cache.put("code_graph_query", params, result)
    assert cache.get("code_graph_query", params) is not None

    cache.invalidate("code_graph_query")
    assert cache.get("code_graph_query", params) is None


def test_graph_revision_isolates_entries(tmp_path):
    cache = GraphCache(cache_dir=tmp_path, ttl_minutes=30)
    params = {"q": "same"}
    cache.put("code_graph_query", params, "rev1_result", graph_revision="rev1")
    cache.put("code_graph_query", params, "rev2_result", graph_revision="rev2")

    h1 = cache.get("code_graph_query", params, graph_revision="rev1")
    h2 = cache.get("code_graph_query", params, graph_revision="rev2")
    assert h1.result == "rev1_result"
    assert h2.result == "rev2_result"
