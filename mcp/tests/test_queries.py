"""Tests for graph.queries — the composable code<->memory query primitives.

Builds a small overlaid graph (2 code files with a CALLS chain, plus findings
that REFERENCE those symbols across 2 investigations) and exercises every
primitive's happy path plus its fail-open contract.
"""

from __future__ import annotations

import pytest

kuzu = pytest.importorskip("kuzu")

from graph.kuzu_store import KuzuStore
from graph import queries as Q


# Symbol ids are "file::Qualname".
SF = "a.py::A.f"
SG = "a.py::A.g"
SH = "a.py::A.h"
SX = "b.py::B.x"


def _build_store(tmp_path) -> KuzuStore:
    store = KuzuStore(str(tmp_path / "graphdb"))
    assert store.available()

    # --- code graph: 2 files, a CALLS chain x -> f -> g -> h ---
    parsed = [
        {
            "file": "a.py", "lang": "py",
            "symbols": [
                {"id": SF, "name": "f", "kind": "function", "line": 1, "lang": "py", "file": "a.py"},
                {"id": SG, "name": "g", "kind": "function", "line": 2, "lang": "py", "file": "a.py"},
                {"id": SH, "name": "h", "kind": "function", "line": 3, "lang": "py", "file": "a.py"},
            ],
            "edges": [
                {"src": SF, "dst": SG, "type": "call"},
                {"src": SG, "dst": SH, "type": "call"},
            ],
        },
        {
            "file": "b.py", "lang": "py",
            "symbols": [
                {"id": SX, "name": "x", "kind": "function", "line": 1, "lang": "py", "file": "b.py"},
            ],
            "edges": [
                {"src": SX, "dst": SF, "type": "call"},
            ],
        },
    ]
    ing = store.ingest_code(parsed)
    assert ing["symbols"] == 4 and ing["calls"] == 3

    # --- investigation graph: findings across 2 investigations ---
    store.upsert_investigation("inv1", "Investigation One")
    store.upsert_investigation("inv2", "Investigation Two")

    findings = [
        {"id": "fA", "investigation": "inv1", "ftype": "note", "text": "touches f", "confidence": "high", "source": "t", "ts": 1},
        {"id": "fB", "investigation": "inv1", "ftype": "note", "text": "touches g", "confidence": "high", "source": "t", "ts": 2},
        {"id": "fC", "investigation": "inv2", "ftype": "note", "text": "touches h", "confidence": "high", "source": "t", "ts": 3},
        {"id": "fD", "investigation": "inv2", "ftype": "note", "text": "also touches f", "confidence": "high", "source": "t", "ts": 4},
    ]
    for f in findings:
        assert store.upsert_finding(f)

    # REFERENCES: fA->f, fD->f (co-reference), fB->g, fC->h
    assert store.link_references("fA", [SF])
    assert store.link_references("fD", [SF])
    assert store.link_references("fB", [SG])
    assert store.link_references("fC", [SH])
    return store


def test_subgraph_returns_anchor_and_neighbors(tmp_path):
    store = _build_store(tmp_path)
    sg = Q.subgraph(store, "CodeSymbol", SG, hops=1)
    keys = {n["key"] for n in sg["nodes"]}
    # anchor present
    assert SG in keys
    # 1-hop neighbours over all rel types: caller f (CALLS), callee h (CALLS),
    # defining file a.py (DEFINES), referencing finding fB (REFERENCES).
    assert SF in keys and SH in keys
    assert "a.py" in keys and "fB" in keys
    # every node carries a label + props, and there is at least one edge
    anchor = next(n for n in sg["nodes"] if n["key"] == SG)
    assert anchor["label"] == "CodeSymbol"
    assert anchor["props"].get("name") == "g"
    assert sg["edges"]
    for e in sg["edges"]:
        assert e["from"] is not None and e["to"] is not None and e["rel"]


def test_symbol_findings_and_finding_symbols_roundtrip(tmp_path):
    store = _build_store(tmp_path)

    # by id
    fids = {f["id"] for f in Q.symbol_findings(store, SF)}
    assert fids == {"fA", "fD"}
    # by name resolves to the same symbol
    fids_by_name = {f["id"] for f in Q.symbol_findings(store, "f")}
    assert fids_by_name == {"fA", "fD"}

    syms = Q.finding_symbols(store, "fA")
    assert [s["id"] for s in syms] == [SF]
    assert syms[0]["name"] == "f"


def test_investigation_footprint_lists_referenced_symbols(tmp_path):
    store = _build_store(tmp_path)
    fp = Q.investigation_footprint(store, "inv1")
    assert fp["finding_count"] == 2
    sym_ids = {s["id"] for s in fp["symbols"]}
    assert sym_ids == {SF, SG}
    assert fp["files"] == ["a.py"]


def test_symbol_impact_includes_transitive_caller_and_finding(tmp_path):
    store = _build_store(tmp_path)
    imp = Q.symbol_impact(store, SH, hops=3)
    caller_ids = {c["id"] for c in imp["callers"]}
    # direct caller g, transitive callers f and x (x -> f -> g -> h)
    assert SG in caller_ids
    assert SF in caller_ids  # transitive
    assert SX in caller_ids  # transitive across files
    assert SH not in caller_ids  # target itself excluded from callers
    # findings referencing the symbol (fC references h)
    finding_ids = {f["id"] for f in imp["findings"]}
    assert "fC" in finding_ids
    # investigations of those findings
    inv_ids = {i["id"] for i in imp["investigations"]}
    assert "inv2" in inv_ids


def test_related_findings_via_code_finds_coreferencer(tmp_path):
    store = _build_store(tmp_path)
    rel = Q.related_findings_via_code(store, "fA")
    ids = {f["id"] for f in rel}
    # fD co-references SF with fA; fB/fC reference different symbols
    assert ids == {"fD"}
    assert rel[0]["shared"] == 1


def test_primitives_fail_open_on_unavailable_store():
    class _Dead:
        def available(self):
            return False

        def code_query(self, *a, **k):
            raise AssertionError("should not be reached when unavailable")

    ks = _Dead()
    assert Q.subgraph(ks, "CodeSymbol", SG) == {"nodes": [], "edges": []}
    assert Q.symbol_findings(ks, SF) == []
    assert Q.finding_symbols(ks, "fA") == []
    assert Q.investigation_footprint(ks, "inv1") == {"symbols": [], "files": [], "finding_count": 0}
    assert Q.symbol_impact(ks, SH) == {"callers": [], "findings": [], "investigations": []}
    assert Q.related_findings_via_code(ks, "fA") == []
    # unknown anchor label is rejected fail-open even if the store were live
    assert Q.subgraph(ks, "Bogus", "x") == {"nodes": [], "edges": []}
