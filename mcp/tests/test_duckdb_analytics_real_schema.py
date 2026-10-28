"""DuckDB analytics against the schema real findings.jsonl files have.

test_duckdb_analytics.py seeds ``created_at`` strings on every row, which no real finding has:
real rows carry ``created_at_ts`` (epoch seconds) and ``ts``, UUID-looking ids, and the files
also hold ``record_type: "access"`` rows. Against the real memory dir the search raised a Binder
Error (``created_at`` not found), then returned UUID objects json.dumps cannot encode, and the
stats counted the access rows as findings. Each test here fails on that code.

Mutation log (each run against the fixed code, reverting one part):
- project ``created_at`` unconditionally: every test using this fixture raised Binder Error
- drop the record_type exclusion: test_access_rows_are_not_findings, test_stats_exclude_access_rows red
- drop the VARCHAR cast on id: test_results_are_json_serialisable red
- drop the created_at_ts branch: test_days_filter_uses_created_at_ts red
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import uuid
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

pytest.importorskip("duckdb")
from loci.analytics import cross_investigation_search, cross_investigation_stats  # noqa: E402

UTC = dt.timezone.utc
NOW = dt.datetime.now(UTC)


def _epoch(when: dt.datetime) -> int:
    return int(when.timestamp())


def _write(root: Path, inv: str, rows: list[dict]) -> None:
    d = root / inv
    d.mkdir(parents=True)
    with open(d / "findings.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _finding(fid, text, *, when, nc=0.5, **extra):
    row = {"id": fid, "type": "observed", "text": text, "confidence": "medium",
           "numeric_confidence": nc, "created_at_ts": _epoch(when),
           "ts": when.isoformat(), "resolution": "open"}
    row.update(extra)
    return row


ACCESS_ID = "5b0a0b0e-0000-4000-8000-00000000acc1"
ACCESS = {"id": ACCESS_ID, "investigation_id": "inv-a", "last_accessed": "2026-10-17T00:00:00+00:00",
          "query": "something", "record_type": "access"}


@pytest.fixture
def real(tmp_path):
    """Two investigations in the real shape: no created_at, no source, UUID ids, access rows."""
    recent = NOW - dt.timedelta(hours=2)
    old = NOW - dt.timedelta(days=400)
    ids = [str(uuid.uuid4()) for _ in range(3)]
    _write(tmp_path, "inv-a", [
        _finding(ids[0], "recent opnsense finding", when=recent, nc=0.9),
        _finding(ids[1], "old firewall finding", when=old, nc=0.4, type="inferred"),
        ACCESS,
    ])
    _write(tmp_path, "inv-b", [
        _finding(ids[2], "recent qdrant finding", when=recent, nc=0.7),
        ACCESS,
    ])
    return tmp_path, ids, recent


def test_search_runs_without_a_created_at_column(real):
    root, ids, _ = real
    rows = cross_investigation_search(memory_dir=root, limit=50)
    assert sorted(r["id"] for r in rows) == sorted(ids)


def test_access_rows_are_not_findings(real):
    root, ids, _ = real
    rows = cross_investigation_search(memory_dir=root, limit=50)
    assert len(rows) == 3
    assert ACCESS_ID not in {r["id"] for r in rows}


def test_created_at_is_the_utc_iso_time_from_created_at_ts(real):
    root, ids, recent = real
    rows = {r["id"]: r for r in cross_investigation_search(memory_dir=root, limit=50)}
    assert rows[ids[0]]["created_at"] == recent.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_days_filter_uses_created_at_ts(real):
    root, ids, _ = real
    recent_only = cross_investigation_search(days=30, memory_dir=root, limit=50)
    assert sorted(r["id"] for r in recent_only) == sorted([ids[0], ids[2]])
    everything = cross_investigation_search(days=1000, memory_dir=root, limit=50)
    assert sorted(r["id"] for r in everything) == sorted(ids)


def test_epoch_seconds_alone_drive_created_at_and_the_days_filter(tmp_path):
    recent = NOW - dt.timedelta(hours=2)
    old = NOW - dt.timedelta(days=400)
    _write(tmp_path, "inv-x", [{"id": "new", "text": "t", "created_at_ts": _epoch(recent)},
                               {"id": "old", "text": "t", "created_at_ts": _epoch(old)}])
    rows = cross_investigation_search(days=30, memory_dir=tmp_path)
    assert [r["id"] for r in rows] == ["new"]
    assert rows[0]["created_at"] == recent.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_iso_ts_alone_is_the_fallback_when_there_is_no_epoch_or_created_at(tmp_path):
    recent = NOW - dt.timedelta(hours=2)
    old = NOW - dt.timedelta(days=400)
    _write(tmp_path, "inv-x", [{"id": "new", "text": "t", "ts": recent.isoformat()},
                               {"id": "old", "text": "t", "ts": old.isoformat()}])
    rows = cross_investigation_search(days=30, memory_dir=tmp_path)
    assert [r["id"] for r in rows] == ["new"]
    assert rows[0]["created_at"] == recent.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_a_legacy_created_at_column_still_drives_the_days_filter(tmp_path):
    recent = NOW - dt.timedelta(hours=2)
    old = NOW - dt.timedelta(days=400)
    _write(tmp_path, "inv-x", [
        {"id": "new", "type": "observed", "text": "t", "numeric_confidence": 0.5,
         "created_at": recent.strftime("%Y-%m-%dT%H:%M:%SZ")},
        {"id": "old", "type": "observed", "text": "t", "numeric_confidence": 0.5,
         "created_at": old.strftime("%Y-%m-%dT%H:%M:%SZ")},
    ])
    rows = cross_investigation_search(days=30, memory_dir=tmp_path, limit=50)
    assert [r["id"] for r in rows] == ["new"]
    assert rows[0]["created_at"] == recent.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_absent_optional_columns_come_back_as_none(real):
    root, ids, _ = real
    r = cross_investigation_search(query_text="qdrant", memory_dir=root, limit=5)[0]
    assert r["id"] == ids[2]
    assert r["source"] is None
    assert r["resolution"] == "open"
    assert r["text"] == "recent qdrant finding"


def test_results_are_json_serialisable(real):
    root, ids, _ = real
    rows = cross_investigation_search(memory_dir=root, limit=50)
    decoded = json.loads(json.dumps(rows))
    assert sorted(r["id"] for r in decoded) == sorted(ids)
    assert all(isinstance(r["id"], str) for r in rows)


def test_text_and_confidence_filters_on_the_real_shape(real):
    root, ids, _ = real
    assert [r["id"] for r in cross_investigation_search(query_text="firewall", memory_dir=root)] == [ids[1]]
    assert [r["id"] for r in cross_investigation_search(min_confidence=0.8, memory_dir=root)] == [ids[0]]
    assert [r["id"] for r in cross_investigation_search(finding_type="inferred", memory_dir=root)] == [ids[1]]


def test_a_filter_on_a_column_no_file_has_matches_nothing(tmp_path):
    # no `type`, no `numeric_confidence`, no time field in any row
    _write(tmp_path, "inv-x", [{"id": "a", "text": "plain finding"}])
    assert cross_investigation_search(memory_dir=tmp_path)[0]["id"] == "a"          # positive twin
    assert cross_investigation_search(finding_type="observed", memory_dir=tmp_path) == []
    assert cross_investigation_search(min_confidence=0.1, memory_dir=tmp_path) == []
    assert cross_investigation_search(days=30, memory_dir=tmp_path) == []


def test_stats_exclude_access_rows(real):
    root, ids, _ = real
    s = cross_investigation_stats(memory_dir=root)
    assert s["total_findings"] == 3
    assert s["investigation_count"] == 2
    assert s["open_count"] == 3
    assert s["resolved_count"] == 0
    assert s["avg_confidence"] == pytest.approx((0.9 + 0.4 + 0.7) / 3)


def test_stats_count_every_row_when_no_file_has_a_record_type(tmp_path):
    _write(tmp_path, "inv-x", [{"id": "a", "text": "t", "numeric_confidence": 0.5, "resolution": "open"},
                               {"id": "b", "text": "t", "numeric_confidence": 0.5, "resolution": "resolved"}])
    s = cross_investigation_stats(memory_dir=tmp_path)
    assert (s["total_findings"], s["open_count"], s["resolved_count"]) == (2, 1, 1)


def test_stats_without_confidence_or_resolution_columns_do_not_error(tmp_path):
    _write(tmp_path, "inv-x", [{"id": "a", "text": "t"}])
    s = cross_investigation_stats(memory_dir=tmp_path)
    assert s == {"total_findings": 1, "investigation_count": 1, "avg_confidence": None,
                 "open_count": 0, "resolved_count": 0}


def test_empty_memory_dir_is_empty_not_an_error(tmp_path):
    assert cross_investigation_search(memory_dir=tmp_path) == []
    assert cross_investigation_stats(memory_dir=tmp_path) == {"total_findings": 0, "investigation_count": 0}
