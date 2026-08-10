"""End-to-end integration tests for the LadybugDB + tree-sitter graph migration.

Drives the server tool functions in-process (no MCP transport needed) against a
temp memory dir, verifying findings mirror into the graph and that the
entity-lookup / related-cases / contamination / code-graph paths are graph-backed.
"""
import json
import pytest

import server as S


@pytest.fixture
def srv(tmp_path, monkeypatch):
    """Point the server at an isolated temp memory dir + reset the LadybugDB singleton."""
    monkeypatch.setattr(S, "MEMORY_DIR", tmp_path / "mem")
    monkeypatch.setattr(S, "_ladybug_store", None, raising=False)
    monkeypatch.setattr(S, "_ladybug_failed", False, raising=False)
    return S


def _store(S, inv, ftype, text, source, conf, derived_from=None):
    return json.loads(S.investigation_store(inv, ftype, text, source, conf,
                                            derived_from=derived_from))["finding_id"]


def test_findings_mirror_into_graph(srv):
    S = srv
    S.investigation_start("inv-a", "Case A")
    f1 = _store(S, "inv-a", "observed", "malicious beacon to 203.0.113.9", "edr", "high")
    _store(S, "inv-a", "inferred", "host 203.0.113.9 again", "edr", "medium", derived_from=f1)
    ks = S._get_ladybug()
    assert ks is not None
    assert ks.code_query("MATCH (f:Finding) RETURN count(f)")[0][0] == 2
    assert ks.code_query(
        "MATCH (:Finding)-[:MENTIONS]->(e:Entity {name:'203.0.113.9'}) RETURN count(*)")[0][0] == 2
    assert ks.code_query("MATCH (:Finding)-[:DERIVED_FROM]->(:Finding) RETURN count(*)")[0][0] == 1
    assert ks.code_query("MATCH (i:Investigation) RETURN count(i)")[0][0] == 1


def test_entity_lookup_is_graph_primary_and_cross_case(srv):
    S = srv
    S.investigation_start("inv-a", "A")
    _store(S, "inv-a", "observed", "beacon 203.0.113.9", "edr", "high")
    S.investigation_start("inv-b", "B")
    _store(S, "inv-b", "observed", "prior sighting of 203.0.113.9", "edr", "high")
    el = json.loads(S.investigation_entity_lookup("203.0.113.9"))
    assert el["retrieval"] == "ladybug"
    assert el["total_findings"] == 2
    assert el["investigations_count"] == 2  # cross-case


def test_related_cases_graph_primary(srv):
    S = srv
    S.investigation_start("inv-a", "A")
    _store(S, "inv-a", "observed", "evil.example.com resolved", "dns", "high")
    S.investigation_start("inv-b", "B")
    _store(S, "inv-b", "observed", "callback to evil.example.com", "proxy", "high")
    rc = json.loads(S.investigation_related_cases("evil.example.com"))["results"][0]
    assert rc["retrieval"] == "ladybug"
    assert rc["related_investigation_count"] >= 1


def test_contamination_via_graph_matches_reference(srv):
    S = srv
    S.investigation_start("inv-a", "A")
    seed = _store(S, "inv-a", "observed", "beacon 198.51.100.7", "edr", "high")
    child = _store(S, "inv-a", "inferred", "escalation note", "an", "medium", derived_from=seed)
    S.investigation_start("inv-b", "B")
    cross = _store(S, "inv-b", "observed", "same 198.51.100.7 elsewhere", "edr", "high")
    ks = S._get_ladybug()
    out = ks.contamination([seed])
    ids = set(out["contaminated_ids"])
    assert seed in ids and child in ids and cross in ids  # derived + cross-case entity
    assert out["reasons"][child] == ["derived_from:" + seed] or "derived_from:" + seed in out["reasons"][child]
    assert any(r.startswith("entity:198.51.100.7") for r in out["reasons"][cross])


def test_code_graph_ingest_and_query(srv):
    S = srv
    ing = json.loads(S.code_graph_ingest("graph/ladybug_store.py"))
    assert ing["ingested"]["symbols"] > 0 and ing["ingested"]["calls"] > 0
    q = json.loads(S.code_graph_query(
        "MATCH (c:CodeSymbol)-[:CALLS]->(t:CodeSymbol) RETURN c.name, t.name LIMIT 3"))
    assert q["row_count"] > 0
    # write-guard must reject mutating cypher
    bad = json.loads(S.code_graph_query("MATCH (n) DETACH DELETE n"))
    assert "error" in bad and "read-only" in bad["error"]


def test_backfill_of_preexisting_findings(srv, tmp_path):
    """A fresh graph backfills findings already on disk (the reconnect path)."""
    S = srv
    # write findings WITHOUT the graph (simulate pre-existing on-disk state)
    S.investigation_start("inv-old", "Old case")
    _store(S, "inv-old", "observed", "old beacon 203.0.113.99", "edr", "high")
    _store(S, "inv-old", "observed", "old beacon 203.0.113.99 again", "edr", "high")
    # drop the graph + its file, reset singleton -> next _get_ladybug triggers backfill
    import shutil
    ks = S._get_ladybug()
    graph_path = tmp_path / "mem" / "graph.ladybug"
    del ks
    S._ladybug_store = None
    S._ladybug_failed = False
    if graph_path.exists():
        shutil.rmtree(graph_path, ignore_errors=True)
    ks2 = S._get_ladybug()  # empty graph -> backfill runs
    assert ks2.code_query("MATCH (f:Finding) RETURN count(f)")[0][0] == 2


def test_relink_invalidates_the_symbol_index_cache(tmp_path, monkeypatch):
    """code_memory_relink must actually drop the cached symbol index.

    Regression: graph_tools.code_memory_relink declared
    `global _symbol_index_cache, _symbol_index_count` and assigned None/-1. Those
    names are not defined in graph_tools, so the `global` bound them in
    graph_tools' OWN namespace and the real cache -- which lives in ladybug_ops --
    was never touched. Auto-linking after a relink kept using a stale index until
    the CodeSymbol count happened to change; a relink that rewires edges without
    changing the symbol count never triggered a rebuild, producing wrong or
    missing REFERENCES edges on findings stored afterwards.
    """
    import graph_tools
    import ladybug_ops

    # Prime the cache with a recognisable value.
    ladybug_ops._symbol_index_cache = {"sentinel": True}
    ladybug_ops._symbol_index_count = 7

    # Force relink_all to succeed without needing a live graph.
    class _FakeLinker:
        @staticmethod
        def relink_all(ks):
            return {"relinked": 0}

    monkeypatch.setattr(graph_tools, "_get_ladybug", lambda: object(), raising=False)
    import sys, types
    fake_graph = types.ModuleType("graph")
    fake_graph.linker = _FakeLinker
    monkeypatch.setitem(sys.modules, "graph.linker", _FakeLinker)

    graph_tools.code_memory_relink()

    assert ladybug_ops._symbol_index_cache is None, (
        "relink did not clear the real cache — the invalidation wrote to the "
        "wrong namespace"
    )
    assert ladybug_ops._symbol_index_count == -1
