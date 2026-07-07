"""Tests for graph.kuzu_store.KuzuStore.

Verifies the Kuzu graph port of ``find_contamination`` is byte-for-byte
equivalent to the in-memory reference, plus code ingestion and the read APIs.
"""

from __future__ import annotations

import pytest

kuzu = pytest.importorskip("kuzu")

from graph.kuzu_store import KuzuStore
from memcheck.checks.contagion import find_contamination


# The single distinctive entity shared between the seed and f2.
URL = "http://localhost:8080/v1/foo"


def _entities_of(text: str) -> dict:
    """Reference entity extractor mirroring what we store in the graph.

    ``urls`` is a distinctive bucket; ``words`` is not (so a shared common word
    never anchors contamination) — exactly the split the graph encodes via the
    Entity.distinctive flag.
    """
    out = {"urls": [], "words": []}
    if URL in text:
        out["urls"].append(URL)
    if "commonword" in text:
        out["words"].append("commonword")
    return out


# In-memory findings for the reference algorithm. Supplied in id-sorted order so
# the reference's dict-insertion iteration matches the store's sorted iteration.
MEM_FINDINGS = [
    {"id": "f1", "text": f"seed references {URL}"},
    {"id": "f2", "text": f"another mention of {URL} here"},
    {"id": "f3", "text": "built on f2", "derived_from": "f2"},
    {"id": "f4", "text": "built on f3", "derived_from": "f3"},
    {"id": "f5", "text": "semantically near the seed"},
    {"id": "f6", "text": "shares only a commonword"},
]


def _build_store(tmp_path) -> KuzuStore:
    store = KuzuStore(str(tmp_path / "graphdb"))
    assert store.available()

    store.upsert_investigation("invA", "Investigation A")
    store.upsert_investigation("invB", "Investigation B")

    # Findings (f1/f2 in invA, the rest carry no investigation except f2->invB below)
    for f in MEM_FINDINGS:
        rec = {"id": f["id"], "text": f["text"], "investigation": "invA",
               "ftype": "note", "confidence": "high", "source": "test", "ts": 1}
        store.upsert_finding(rec)

    # Distinctive URL entity shared by f1 and f2; f6 shares only a NON-distinctive word.
    store.link_mentions("f1", [(URL, "url", True)])
    store.link_mentions("f2", [(URL, "url", True)])
    store.link_mentions("f6", [("commonword", "word", False)])

    # Derivation chain: f4 -> f3 -> f2 (f2 is entity-contaminated).
    store.link_derived_from("f3", ["f2"])
    store.link_derived_from("f4", ["f3"])
    return store


def test_contamination_matches_reference(tmp_path):
    store = _build_store(tmp_path)

    graph_res = store.contamination(["f1"], semantic_neighbor_ids=["f5"])
    ref_res = find_contamination(
        ["f1"], MEM_FINDINGS,
        entities_of=_entities_of, semantic_neighbor_ids=["f5"],
    )

    # Identical contaminated set (and ordering), identical reasons dict.
    assert graph_res["contaminated_ids"] == ref_res["contaminated_ids"]
    assert graph_res["reasons"] == ref_res["reasons"]

    # Sanity: the expected shape.
    assert graph_res["contaminated_ids"] == ["f1", "f2", "f3", "f4", "f5"]
    assert graph_res["reasons"]["f1"] == ["seed"]
    assert graph_res["reasons"]["f5"] == ["semantic"]
    assert graph_res["reasons"]["f2"] == [f"entity:{URL}"]
    assert graph_res["reasons"]["f3"] == ["derived_from:f2"]
    assert graph_res["reasons"]["f4"] == ["derived_from:f3"]
    assert "f6" not in graph_res["reasons"]


def test_contamination_threshold(tmp_path):
    store = _build_store(tmp_path)
    # Require 2 shared distinctive entities — f2 only shares 1, so no entity hit.
    res = store.contamination(["f1"], min_shared_entities=2)
    assert res["contaminated_ids"] == ["f1"]
    assert res["reasons"] == {"f1": ["seed"]}


def test_entity_findings(tmp_path):
    store = _build_store(tmp_path)
    hits = {f["id"] for f in store.entity_findings("HTTP://LOCALHOST:8080/V1/FOO")}
    assert hits == {"f1", "f2"}  # case-insensitive name match


def test_related_investigations(tmp_path):
    store = _build_store(tmp_path)
    # Move f2 into invB so invA and invB share the URL entity.
    store.upsert_finding({"id": "f2", "text": f"mention {URL}", "investigation": "invB",
                          "ftype": "note", "confidence": "high", "source": "t", "ts": 1})
    related = store.related_investigations("invA")
    ids = [r["id"] for r in related]
    assert "invB" in ids
    b = next(r for r in related if r["id"] == "invB")
    assert b["shared"] >= 1
    assert b["title"] == "Investigation B"


def test_ingest_code_and_reads(tmp_path):
    store = KuzuStore(str(tmp_path / "codedb"))
    assert store.available()

    parsed = [
        {
            "file": "app/main.py", "lang": "python",
            "symbols": [
                {"id": "app/main.py::main", "name": "main", "kind": "function",
                 "line": 10, "lang": "python", "file": "app/main.py"},
                {"id": "app/main.py::helper", "name": "helper", "kind": "function",
                 "line": 2, "lang": "python", "file": "app/main.py"},
            ],
            # DEFINES + IMPORTS also appear in edges (as real code_parse emits);
            # ingest ignores them here (DEFINES from symbols, IMPORTS from list).
            "edges": [
                {"src": "app/main.py", "dst": "app/main.py::main", "type": "DEFINES"},
                {"src": "app/main.py::main", "dst": "helper", "type": "CALLS"},
                {"src": "app/main.py", "dst": "os", "type": "IMPORTS"},
            ],
            "imports": ["os"],
        },
    ]
    counts = store.ingest_code(parsed)
    assert counts["files"] == 1
    assert counts["symbols"] == 2
    assert counts["defines"] == 2
    assert counts["calls"] == 1       # resolved bare 'helper' -> symbol id
    assert counts["imports"] == 1

    # callers_of resolves the CALLS graph by callee name.
    callers = store.callers_of("helper")
    assert [c["id"] for c in callers] == ["app/main.py::main"]

    # Finding -> CodeSymbol REFERENCES, queried by both id and name.
    store.upsert_finding({"id": "cf1", "text": "notes about helper", "investigation": "invX"})
    store.link_references("cf1", ["app/main.py::helper"])
    assert [f["id"] for f in store.symbol_findings("app/main.py::helper")] == ["cf1"]
    assert [f["id"] for f in store.symbol_findings("helper")] == ["cf1"]


def test_ingest_code_type_aware_calls(tmp_path):
    """Type-aware CALLS resolution: receiver'd calls are never resolved by bare
    global name; external-import receivers (Log.w) are dropped entirely."""
    store = KuzuStore(str(tmp_path / "typedb"))
    assert store.available()

    parsed = [
        {
            "file": "app/A.java", "lang": "java",
            "import_map": {"Log": "android.util.Log"},
            "symbols": [
                {"id": "app/A.java::AppClass", "name": "AppClass", "kind": "class",
                 "line": 1, "lang": "java", "file": "app/A.java"},
                {"id": "app/A.java::AppClass.caller", "name": "caller", "kind": "method",
                 "line": 5, "lang": "java", "file": "app/A.java"},
                {"id": "app/A.java::AppClass.foo", "name": "foo", "kind": "method",
                 "line": 10, "lang": "java", "file": "app/A.java"},
                {"id": "app/A.java::AppClass.bar", "name": "bar", "kind": "method",
                 "line": 15, "lang": "java", "file": "app/A.java"},
                {"id": "app/A.java::AppClass.helper", "name": "helper", "kind": "method",
                 "line": 20, "lang": "java", "file": "app/A.java"},
            ],
            "edges": [
                # External import receiver -> DROP (no edge).
                {"src": "app/A.java::AppClass.caller", "dst": "w",
                 "type": "CALLS", "receiver": "Log", "recv_kind": "name"},
                # this.foo() -> same class's foo.
                {"src": "app/A.java::AppClass.caller", "dst": "foo",
                 "type": "CALLS", "receiver": "this", "recv_kind": "self"},
                # AppClass.bar() -> app-type static call resolves to AppClass.bar.
                {"src": "app/A.java::AppClass.caller", "dst": "bar",
                 "type": "CALLS", "receiver": "AppClass", "recv_kind": "name"},
                # bare helper() -> enclosing class method.
                {"src": "app/A.java::AppClass.caller", "dst": "helper",
                 "type": "CALLS", "receiver": None, "recv_kind": "none"},
            ],
            "imports": ["android.util.Log"],
        },
    ]
    counts = store.ingest_code(parsed)
    assert counts["calls"] == 3
    assert counts["calls_dropped_external"] == 1

    # Log.w produced NO edge.
    assert store.callers_of("w") == []
    # this.foo / AppClass.bar / bare helper all resolve to the caller.
    for name in ("foo", "bar", "helper"):
        assert [c["id"] for c in store.callers_of(name)] == ["app/A.java::AppClass.caller"], name


def test_ingest_code_drops_untyped_and_expr_receivers(tmp_path):
    """Untyped variable receivers and complex-expression receivers are dropped
    in v1 (no global by-name fallback)."""
    store = KuzuStore(str(tmp_path / "untypeddb"))
    assert store.available()

    parsed = [
        {
            "file": "app/B.java", "lang": "java", "import_map": {},
            "symbols": [
                {"id": "app/B.java::C", "name": "C", "kind": "class",
                 "line": 1, "lang": "java", "file": "app/B.java"},
                {"id": "app/B.java::C.run", "name": "run", "kind": "method",
                 "line": 5, "lang": "java", "file": "app/B.java"},
                # A globally-unique 'target' that a receiver'd call must NOT reach.
                {"id": "app/B.java::C.target", "name": "target", "kind": "method",
                 "line": 9, "lang": "java", "file": "app/B.java"},
            ],
            "edges": [
                # Unknown variable receiver -> DROP (no by-name fallback).
                {"src": "app/B.java::C.run", "dst": "target",
                 "type": "CALLS", "receiver": "obj", "recv_kind": "name"},
                # Complex expression receiver -> DROP.
                {"src": "app/B.java::C.run", "dst": "target",
                 "type": "CALLS", "receiver": "a.b().c", "recv_kind": "expr"},
            ],
            "imports": [],
        },
    ]
    counts = store.ingest_code(parsed)
    assert counts["calls"] == 0
    assert counts["calls_dropped_unresolved"] == 2
    assert store.callers_of("target") == []


def test_code_query_write_guard(tmp_path):
    store = KuzuStore(str(tmp_path / "guarddb"))
    assert store.available()
    store.ingest_code([{
        "file": "x.py", "lang": "python",
        "symbols": [{"id": "x.py::f", "name": "f", "kind": "function",
                     "line": 1, "lang": "python", "file": "x.py"}],
        "edges": [], "imports": [],
    }])

    # Read-only query works.
    rows = store.code_query("MATCH (s:CodeSymbol) RETURN s.id")
    assert ["x.py::f"] in rows

    for bad in [
        "CREATE NODE TABLE Z(id STRING PRIMARY KEY)",
        "MATCH (n) DELETE n",
        "MATCH (s:CodeSymbol) SET s.name='x'",
        "MERGE (n:CodeSymbol {id:'q'})",
        "DROP TABLE CodeSymbol",
    ]:
        with pytest.raises(ValueError):
            store.code_query(bad)


def test_fail_open_on_bad_query(tmp_path):
    store = KuzuStore(str(tmp_path / "faildb"))
    assert store.available()
    # Nonsense read via code_query is swallowed -> [].
    assert store.code_query("MATCH (n:NoSuchTable) RETURN n") == []
    # Unknown entity/symbol -> empty, never raises.
    assert store.entity_findings("nope") == []
    assert store.callers_of("nope") == []
    assert store.symbol_findings("nope") == []
    assert store.contamination(["ghost"]) == {"contaminated_ids": ["ghost"],
                                              "reasons": {"ghost": ["seed"]}}
