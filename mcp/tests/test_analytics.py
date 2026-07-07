"""Tests for graph.analytics — the composed code<->memory tools."""
import pytest

from graph.kuzu_store import KuzuStore
from graph import analytics as A


@pytest.fixture
def ks(tmp_path):
    s = KuzuStore(str(tmp_path / "g.kuzu"))
    assert s.available()
    e = s._exec
    e("MERGE (c:CodeFile {path:'a.java'}) SET c.lang='java'")
    for sid, name, kind in [
        ("a.java::ClassA", "ClassA", "class"),
        ("a.java::ClassA.foo", "foo", "method"),
        ("a.java::ClassA.caller", "caller", "method"),
    ]:
        e("MERGE (s:CodeSymbol {id:$i}) SET s.name=$n, s.kind=$k, s.file='a.java', s.line=1, s.lang='java'",
          {"i": sid, "n": name, "k": kind})
        e("MATCH (c:CodeFile {path:'a.java'}),(s:CodeSymbol {id:$i}) MERGE (c)-[:DEFINES]->(s)", {"i": sid})
    e("MATCH (a:CodeSymbol {id:'a.java::ClassA.caller'}),(b:CodeSymbol {id:'a.java::ClassA.foo'}) "
      "MERGE (a)-[:CALLS]->(b)")
    for fid, inv in [("f1", "inv1"), ("f2", "inv2")]:
        e("MERGE (f:Finding {id:$i}) SET f.investigation=$v, f.text=$t, f.ftype='observed', "
          "f.confidence='high', f.source='x', f.ts=1", {"i": fid, "v": inv, "t": f"foo issue in {inv}"})
        e("MERGE (iv:Investigation {id:$v})", {"v": inv})
        e("MATCH (f:Finding {id:$i}),(iv:Investigation {id:$v}) MERGE (f)-[:IN_INVESTIGATION]->(iv)",
          {"i": fid, "v": inv})
        e("MATCH (f:Finding {id:$i}),(s:CodeSymbol {id:'a.java::ClassA.foo'}) MERGE (f)-[:REFERENCES]->(s)",
          {"i": fid})
    return s


def test_impact_report_method(ks):
    r = A.impact_report(ks, "foo")
    assert "foo" in {x["name"] for x in r["resolved"]}
    assert "caller" in r["direct_callers"]
    assert r["referencing_finding_count"] == 2
    assert {i["id"] for i in r["investigations"]} == {"inv1", "inv2"}


def test_impact_report_class_folds_methods(ks):
    r = A.impact_report(ks, "ClassA")
    # class target folds in ClassA.foo -> the referencing findings still surface.
    assert r["referencing_finding_count"] == 2
    assert "ClassA.foo" in {x["id"].split("::")[-1] for x in r["resolved"]}


def test_finding_code_context(ks):
    ctx = A.finding_code_context(ks, "f1")
    foo = next((s for s in ctx["symbols"] if s["name"] == "foo"), None)
    assert foo is not None
    assert "caller" in foo["callers"]


def test_related_investigations_via_code(ks):
    rel = A.related_investigations_via_code(ks, "inv1")
    assert rel and rel[0]["investigation"] == "inv2"
    assert rel[0]["shared_symbols"] >= 1


def test_fail_open():
    class Dead:
        def available(self):
            return False

    assert A.impact_report(Dead(), "x")["resolved"] == []
    assert A.finding_code_context(Dead(), "x")["symbols"] == []
    assert A.related_investigations_via_code(Dead(), "x") == []
