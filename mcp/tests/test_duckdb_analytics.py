"""Tests for DuckDB cross-investigation analytics (loci.analytics).

Naming: test_duckdb_analytics.py to avoid collision with the graph analytics
tests in test_analytics.py (graph.analytics — code<->memory composed tools).

Mutation log:
- test_cross_investigation_search_all: deleted the SELECT → returned [], asserts len==30 failed ✓
- test_cross_investigation_search_confidence_filter: removed WHERE clause → returned 20 rows, exact count failed ✓
- test_cross_investigation_search_type_filter: removed type filter → returned 10 rows, exact count failed ✓
- test_cross_investigation_search_text_filter: removed ILIKE filter → returned 10, exact count failed ✓
- test_investigation_stats_total: returned wrong count → failed ✓
- test_investigation_stats_investigation_count: returned wrong count → failed ✓
- test_duckdb_not_available_raises: removed the _DUCKDB_AVAILABLE guard → no ImportError, failed ✓
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure loci package is importable when tests run from mcp/
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _seed_investigations(root: Path, n_investigations: int = 3, n_findings: int = 10) -> Path:
    """Write fake investigation JSONL structure under root."""
    for i in range(n_investigations):
        inv_dir = root / f"inv-{i:04d}"
        inv_dir.mkdir()
        with open(inv_dir / "findings.jsonl", "w") as fh:
            for j in range(n_findings):
                finding = {
                    "id": f"f-{i}-{j}",
                    "type": "observed" if j % 2 == 0 else "inferred",
                    "text": f"Finding {j} in investigation {i}",
                    "confidence": "high" if j > 5 else "medium",
                    "numeric_confidence": 0.9 if j > 5 else 0.5,
                    "created_at": "2026-10-17T18:00:00Z",
                    "resolution": "open",
                    "source": "test",
                }
                fh.write(json.dumps(finding) + "\n")
    return root


# ---------------------------------------------------------------------------
# cross_investigation_search — positive paths
# ---------------------------------------------------------------------------

def test_cross_investigation_search_all(tmp_path):
    """All 30 findings (3 investigations × 10) are returned with no filters."""
    pytest.importorskip("duckdb")
    from loci.analytics import cross_investigation_search

    _seed_investigations(tmp_path, n_investigations=3, n_findings=10)
    results = cross_investigation_search(memory_dir=tmp_path, limit=100)

    assert len(results) == 30
    inv_ids = {r["investigation_id"] for r in results}
    assert inv_ids == {"inv-0000", "inv-0001", "inv-0002"}


def test_cross_investigation_search_returns_expected_fields(tmp_path):
    """Each result carries the fields the callers depend on."""
    pytest.importorskip("duckdb")
    from loci.analytics import cross_investigation_search

    _seed_investigations(tmp_path, n_investigations=1, n_findings=1)
    results = cross_investigation_search(memory_dir=tmp_path, limit=10)

    assert len(results) == 1
    r = results[0]
    assert r["id"] == "f-0-0"
    assert r["type"] == "observed"
    assert r["text"] == "Finding 0 in investigation 0"
    assert r["investigation_id"] == "inv-0000"


def test_cross_investigation_search_confidence_filter(tmp_path):
    """min_confidence=0.8 keeps only the 8 findings (4 per inv × 2 invs) with nc=0.9."""
    pytest.importorskip("duckdb")
    from loci.analytics import cross_investigation_search

    # j > 5 → nc=0.9; j <= 5 → nc=0.5.  4 findings per investigation pass.
    _seed_investigations(tmp_path, n_investigations=2, n_findings=10)
    results = cross_investigation_search(min_confidence=0.8, memory_dir=tmp_path, limit=100)

    assert len(results) == 8
    assert all(r["numeric_confidence"] == pytest.approx(0.9) for r in results)


def test_cross_investigation_search_type_filter(tmp_path):
    """finding_type='observed' returns exactly 10 findings (j%2==0 → 5 per inv × 2 invs)."""
    pytest.importorskip("duckdb")
    from loci.analytics import cross_investigation_search

    _seed_investigations(tmp_path, n_investigations=2, n_findings=10)
    results = cross_investigation_search(finding_type="observed", memory_dir=tmp_path, limit=100)

    assert len(results) == 10
    assert all(r["type"] == "observed" for r in results)


def test_cross_investigation_search_text_filter(tmp_path):
    """query_text='investigation 0' matches all 10 findings from inv-0000 only."""
    pytest.importorskip("duckdb")
    from loci.analytics import cross_investigation_search

    # Text pattern: "Finding J in investigation 0" — the substring "investigation 0"
    # appears only in inv-0000's findings, not in inv-0001's "… investigation 1".
    _seed_investigations(tmp_path, n_investigations=2, n_findings=10)
    results = cross_investigation_search(query_text="investigation 0", memory_dir=tmp_path, limit=100)

    assert len(results) == 10
    assert all(r["investigation_id"] == "inv-0000" for r in results)


def test_cross_investigation_search_limit_respected(tmp_path):
    """limit=5 caps results at 5 even when more match."""
    pytest.importorskip("duckdb")
    from loci.analytics import cross_investigation_search

    _seed_investigations(tmp_path, n_investigations=3, n_findings=10)
    results = cross_investigation_search(memory_dir=tmp_path, limit=5)

    assert len(results) == 5


# ---------------------------------------------------------------------------
# cross_investigation_search — negative / edge paths (each has a positive twin above)
# ---------------------------------------------------------------------------

def test_no_findings_returns_empty(tmp_path):
    """Empty memory_dir (no findings.jsonl files) returns [] without raising."""
    pytest.importorskip("duckdb")
    from loci.analytics import cross_investigation_search

    # No investigation subdirs created → glob matches nothing.
    results = cross_investigation_search(memory_dir=tmp_path)

    assert results == []


def test_duckdb_not_available_raises(tmp_path):
    """ImportError with 'duckdb' in message when _DUCKDB_AVAILABLE is False.

    Positive twin: test_cross_investigation_search_all verifies the success path.
    """
    pytest.importorskip("duckdb")
    import loci.analytics as a

    orig = a._DUCKDB_AVAILABLE
    try:
        a._DUCKDB_AVAILABLE = False
        with pytest.raises(ImportError, match="duckdb"):
            a.cross_investigation_search(memory_dir=tmp_path)
    finally:
        a._DUCKDB_AVAILABLE = orig


# ---------------------------------------------------------------------------
# cross_investigation_stats — positive paths
# ---------------------------------------------------------------------------

def test_investigation_stats_total(tmp_path):
    """total_findings is exactly 30 (3 investigations × 10 findings each)."""
    pytest.importorskip("duckdb")
    from loci.analytics import cross_investigation_stats

    _seed_investigations(tmp_path, n_investigations=3, n_findings=10)
    stats = cross_investigation_stats(memory_dir=tmp_path)

    assert stats["total_findings"] == 30


def test_investigation_stats_investigation_count(tmp_path):
    """investigation_count is exactly 3 (one findings.jsonl per directory)."""
    pytest.importorskip("duckdb")
    from loci.analytics import cross_investigation_stats

    _seed_investigations(tmp_path, n_investigations=3, n_findings=10)
    stats = cross_investigation_stats(memory_dir=tmp_path)

    assert stats["investigation_count"] == 3


def test_investigation_stats_open_count(tmp_path):
    """All 30 seeded findings have resolution='open', so open_count==30."""
    pytest.importorskip("duckdb")
    from loci.analytics import cross_investigation_stats

    _seed_investigations(tmp_path, n_investigations=3, n_findings=10)
    stats = cross_investigation_stats(memory_dir=tmp_path)

    assert stats["open_count"] == 30
    assert stats["resolved_count"] == 0


def test_investigation_stats_avg_confidence(tmp_path):
    """avg_confidence matches the known mix: 4 high (0.9) + 6 medium (0.5) per inv."""
    pytest.importorskip("duckdb")
    from loci.analytics import cross_investigation_stats

    # Per investigation: j in {6,7,8,9} → nc=0.9 (4 findings); rest → nc=0.5 (6 findings).
    # avg = (4*0.9 + 6*0.5) / 10 = (3.6 + 3.0) / 10 = 0.66
    _seed_investigations(tmp_path, n_investigations=1, n_findings=10)
    stats = cross_investigation_stats(memory_dir=tmp_path)

    assert stats["avg_confidence"] == pytest.approx(0.66, abs=1e-9)


def test_investigation_stats_empty_dir(tmp_path):
    """Empty memory_dir returns zeroed counts without raising."""
    pytest.importorskip("duckdb")
    from loci.analytics import cross_investigation_stats

    stats = cross_investigation_stats(memory_dir=tmp_path)

    assert stats.get("total_findings") == 0
    assert stats.get("investigation_count") == 0


def test_duckdb_stats_not_available_raises(tmp_path):
    """cross_investigation_stats also raises ImportError when _DUCKDB_AVAILABLE is False."""
    pytest.importorskip("duckdb")
    import loci.analytics as a

    orig = a._DUCKDB_AVAILABLE
    try:
        a._DUCKDB_AVAILABLE = False
        with pytest.raises(ImportError, match="duckdb"):
            a.cross_investigation_stats(memory_dir=tmp_path)
    finally:
        a._DUCKDB_AVAILABLE = orig
