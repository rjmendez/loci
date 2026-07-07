"""Tests for graph.linker — the precision Finding -> CodeSymbol linker.

Covers the two things that matter: (1) ``extract_symbol_refs`` fires ONLY on
distinctive evidence (a CamelCase type name, a ``symbol:`` marker, a dotted
type ref, a source-file mention) and never on common words; and (2)
``link_findings`` / ``relink_all`` write ``REFERENCES`` edges for exactly those
and nothing else, idempotently.
"""

from __future__ import annotations

import pytest

kuzu = pytest.importorskip("kuzu")

from graph.kuzu_store import KuzuStore
from graph.linker import (
    build_symbol_index,
    extract_symbol_refs,
    link_findings,
    relink_all,
)

# A distinctive type and a common-named method, ids shaped "file::Qualname".
_FILE = "DeviceMetricsPoller.java"
_TYPE_ID = f"{_FILE}::DeviceMetricsPoller"
_METHOD_ID = f"{_FILE}::DeviceMetricsPoller.getInstance"

_SYMBOLS = [
    {"id": _TYPE_ID, "name": "DeviceMetricsPoller", "kind": "class",
     "file": _FILE, "line": 1, "lang": "java"},
    {"id": _METHOD_ID, "name": "getInstance", "kind": "method",
     "file": _FILE, "line": 10, "lang": "java"},
]


def _index():
    return build_symbol_index(_SYMBOLS)


def _build_store(tmp_path) -> KuzuStore:
    store = KuzuStore(str(tmp_path / "graphdb"))
    assert store.available()
    parsed = [{
        "file": _FILE, "lang": "java",
        "symbols": _SYMBOLS, "edges": [], "imports": [],
    }]
    counts = store.ingest_code(parsed)
    assert counts["symbols"] == 2
    return store


def _references(store: KuzuStore) -> set[tuple[str, str]]:
    rows = store._rows(
        "MATCH (f:Finding)-[:REFERENCES]->(s:CodeSymbol) RETURN f.id, s.id"
    )
    return {(r[0], r[1]) for r in rows}


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
def test_index_shape():
    idx = _index()
    assert idx["by_name"]["DeviceMetricsPoller"] == [_TYPE_ID]
    assert "DeviceMetricsPoller" in idx["type_names"]
    assert "getInstance" not in idx["type_names"]
    # both names are globally unique
    assert {"DeviceMetricsPoller", "getInstance"} <= idx["unique_names"]
    assert _TYPE_ID in idx["file_basenames"][_FILE]


# --------------------------------------------------------------------------- #
# extract_symbol_refs — precision
# --------------------------------------------------------------------------- #
def test_extract_type_name_high():
    refs = extract_symbol_refs("DeviceMetricsPoller crashed on boot", _index())
    assert refs == [(_TYPE_ID, "high")]


def test_extract_common_words_link_nothing():
    # "get" is a real method fragment but a common word; "getInstance" is not
    # present as a token here, so nothing distinctive matches.
    assert extract_symbol_refs("we should get the value", _index()) == []


def test_extract_bare_get_method_never_matches():
    # A finding that only says "get" must not resolve to getInstance.
    assert extract_symbol_refs("please get it and run", _index()) == []


def test_extract_symbol_marker_high():
    refs = extract_symbol_refs("see symbol:getInstance for details", _index())
    assert refs == [(_METHOD_ID, "high")]


def test_extract_dotted_type_member_high():
    refs = extract_symbol_refs(
        "DeviceMetricsPoller.getInstance returned null", _index()
    )
    # both the type (whole-word) and the member (dotted) resolve, HIGH each
    assert (_TYPE_ID, "high") in refs
    assert (_METHOD_ID, "high") in refs


def test_extract_source_file_mention_high():
    refs = dict(extract_symbol_refs("edited DeviceMetricsPoller.java today", _index()))
    # a .java mention links only the file's PRIMARY TYPE, not every member (over-links).
    assert refs.get(_TYPE_ID) == "high"
    assert _METHOD_ID not in refs


def test_extract_unique_camel_identifier_medium():
    idx = build_symbol_index([
        {"id": "x.java::x.computeTrajectory", "name": "computeTrajectory",
         "kind": "method", "file": "x.java", "line": 3, "lang": "java"},
    ])
    refs = extract_symbol_refs("the computeTrajectory step diverged", idx)
    assert refs == [("x.java::x.computeTrajectory", "medium")]


def test_extract_short_identifier_not_medium():
    idx = build_symbol_index([
        {"id": "x.java::x.doIt", "name": "doIt", "kind": "method",
         "file": "x.java", "line": 3, "lang": "java"},
    ])
    # "doIt" is < 8 chars -> never MEDIUM
    assert extract_symbol_refs("call doIt now", idx) == []


def test_extract_ambiguous_plain_name_never():
    # Two symbols share a plain (non-type) name -> ambiguous -> never matched.
    idx = build_symbol_index([
        {"id": "a.java::A.processFrame", "name": "processFrame", "kind": "method",
         "file": "a.java", "line": 1, "lang": "java"},
        {"id": "b.java::B.processFrame", "name": "processFrame", "kind": "method",
         "file": "b.java", "line": 1, "lang": "java"},
    ])
    assert extract_symbol_refs("processFrame was slow", idx) == []


# --------------------------------------------------------------------------- #
# link_findings / relink_all — writes
# --------------------------------------------------------------------------- #
def test_link_findings_only_distinctive(tmp_path):
    store = _build_store(tmp_path)
    findings = [
        {"id": "f_hit", "text": "DeviceMetricsPoller crashed"},
        {"id": "f_miss", "text": "we should get the value"},
    ]
    for f in findings:
        store.upsert_finding({
            "id": f["id"], "text": f["text"], "investigation": "inv",
            "ftype": "note", "confidence": "high", "source": "test", "ts": 1,
        })

    created = link_findings(store, findings)
    assert created == 1

    refs = _references(store)
    assert refs == {("f_hit", _TYPE_ID)}
    # the common-word finding linked to nothing
    assert not any(fid == "f_miss" for fid, _ in refs)


def test_link_findings_idempotent(tmp_path):
    store = _build_store(tmp_path)
    store.upsert_finding({
        "id": "f_hit", "text": "DeviceMetricsPoller crashed", "investigation": "inv",
        "ftype": "note", "confidence": "high", "source": "test", "ts": 1,
    })
    rows = [{"id": "f_hit", "text": "DeviceMetricsPoller crashed"}]
    link_findings(store, rows)
    link_findings(store, rows)  # MERGE -> no duplicate edge
    assert _references(store) == {("f_hit", _TYPE_ID)}


def test_relink_all(tmp_path):
    store = _build_store(tmp_path)
    for fid, text in [
        ("f_hit", "DeviceMetricsPoller crashed"),
        ("f_miss", "we should get the value"),
        ("f_sym", "check symbol:getInstance"),
    ]:
        store.upsert_finding({
            "id": fid, "text": text, "investigation": "inv",
            "ftype": "note", "confidence": "high", "source": "test", "ts": 1,
        })

    result = relink_all(store)
    assert result["findings_scanned"] == 3
    assert result["links_created"] == 2
    assert _references(store) == {("f_hit", _TYPE_ID), ("f_sym", _METHOD_ID)}


def test_fail_open_on_bad_store():
    # A None store and an unavailable store both return zero, never raise.
    assert link_findings(None, [{"id": "x", "text": "DeviceMetricsPoller"}]) == 0
    assert relink_all(None) == {"findings_scanned": 0, "links_created": 0}
