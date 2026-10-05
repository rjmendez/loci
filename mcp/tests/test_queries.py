"""Tests for graph.queries — the composable code<->memory query primitives.

Builds a small overlaid graph (2 code files with a CALLS chain, plus findings
that REFERENCE those symbols across 2 investigations) and exercises every
primitive's happy path plus its fail-open contract.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ladybug")

from graph.ladybug_store import LadybugStore
from graph import queries as Q


# Symbol ids are "file::Qualname".
SF = "a.py::A.f"
SG = "a.py::A.g"
SH = "a.py::A.h"
SX = "b.py::B.x"


def _build_store(tmp_path) -> LadybugStore:
    store = LadybugStore(str(tmp_path / "graphdb"))
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
    # 1-hop neighbours must span all rel types: CALLS both ways, DEFINES, REFERENCES.
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
    # direct caller g, transitive callers f and x (x -> f -> g -> h); h itself excluded
    assert sorted(c["id"] for c in imp["callers"]) == sorted([SG, SF, SX])
    # findings on the target (fC -> h) AND on every caller (fB -> g, fA/fD -> f)
    assert sorted(f["id"] for f in imp["findings"]) == ["fA", "fB", "fC", "fD"]
    assert sorted(i["id"] for i in imp["investigations"]) == ["inv1", "inv2"]
    # hop bound: with hops=1 only the direct caller g (and its finding fB) is reached
    one = Q.symbol_impact(store, SH, hops=1)
    assert [c["id"] for c in one["callers"]] == [SG]
    assert sorted(f["id"] for f in one["findings"]) == ["fB", "fC"]


def test_related_findings_via_code_finds_coreferencer(tmp_path):
    store = _build_store(tmp_path)
    rel = Q.related_findings_via_code(store, "fA")
    ids = {f["id"] for f in rel}
    # fD co-references SF with fA; fB/fC reference different symbols
    assert ids == {"fD"}
    assert rel[0]["shared"] == 1


class _RecordingStore:
    """Store double that records every query (never raises into the fail-open code)."""

    def __init__(self, available):
        self._available = available
        self.queries = []

    def available(self):
        return self._available

    def code_query(self, cypher, params=None):
        self.queries.append(cypher)
        return []


def test_primitives_fail_open_on_unavailable_store():
    ks = _RecordingStore(available=False)
    assert Q.subgraph(ks, "CodeSymbol", SG) == {"nodes": [], "edges": []}
    assert Q.symbol_findings(ks, SF) == []
    assert Q.finding_symbols(ks, "fA") == []
    assert Q.investigation_footprint(ks, "inv1") == {"symbols": [], "files": [], "finding_count": 0}
    assert Q.symbol_impact(ks, SH) == {"callers": [], "findings": [], "investigations": []}
    assert Q.related_findings_via_code(ks, "fA") == []
    assert ks.queries == []          # an unavailable store is never queried


def test_subgraph_rejects_an_unknown_anchor_label_before_querying():
    # The label is interpolated into the Cypher text, so only whitelisted labels
    # may reach the store; a live store must not see the query at all.
    ks = _RecordingStore(available=True)
    for label in ("Bogus", "CodeSymbol) DETACH DELETE (n", ""):
        assert Q.subgraph(ks, label, "x") == {"nodes": [], "edges": []}
    assert ks.queries == []
    # positive twin: a whitelisted label is queried, with the label in the MATCH
    Q.subgraph(ks, "CodeSymbol", SG)
    assert len(ks.queries) == 1 and ks.queries[0].startswith("MATCH (a:CodeSymbol {id:$k})")
