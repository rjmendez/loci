"""analyze/literalaudit.py: producer/consumer table, orphan detection, and
the fenced near-miss pairing — BUG D's regression, against real GraphStore
output over fixtures.py."""
from ..analyze.literalaudit import literal_table, materialize_near_miss_edges, near_miss_pairs, orphans
from ..tests.helpers import build_fixture_store


def _build():
    return build_fixture_store(["literals_produce.py", "literals_consume.py", "literals_keys.py"])


def test_literal_table_matches_notes_jsonl_both_directions():
    store, _, _ = _build()
    groups = {g.match_text: g for g in literal_table(store, flavour="path")}
    g = groups["notes.jsonl"]
    assert len(g.produce_sites) == 1
    assert len(g.consume_sites) == 1
    assert g.produce_sites[0].path == "literals_produce.py"
    assert g.consume_sites[0].path == "literals_consume.py"


def test_orphans_path_finds_bug_d_shape():
    store, _, _ = _build()
    producer_only, consumer_only = orphans(store, flavour="path")
    p_texts = {g.match_text for g in producer_only}
    c_texts = {g.match_text for g in consumer_only}
    assert "graph.ladybug" in p_texts
    assert "graph.kuzu" in c_texts
    # the matched pair must NOT show up as an orphan in either direction
    assert "notes.jsonl" not in p_texts
    assert "notes.jsonl" not in c_texts


def test_orphans_key_finds_dict_get_set_shape():
    store, _, _ = _build()
    producer_only, consumer_only = orphans(store, flavour="key")
    p_texts = {g.match_text for g in producer_only}
    c_texts = {g.match_text for g in consumer_only}
    assert "cfg_write_only" in p_texts
    assert "cfg_lookup_only" in c_texts
    assert "HERMES_PORT" not in p_texts
    assert "HERMES_PORT" not in c_texts


def test_near_miss_pairs_graph_ladybug_and_graph_kuzu():
    store, _, _ = _build()
    producer_only, consumer_only = orphans(store, flavour="path")
    pairs = near_miss_pairs(producer_only, consumer_only)
    matches = [p for p in pairs if p.shared_stem == "graph"]
    assert len(matches) == 1
    nm = matches[0]
    assert {nm.producer.match_text, nm.consumer.match_text} == {"graph.ladybug", "graph.kuzu"}
    assert 0.0 < nm.distance < 1.0


def test_near_miss_requires_cross_module():
    # Two producer-only / consumer-only literals sharing a stem but BOTH
    # sited only in the SAME module must not pair — the design's own
    # "module boundary" requirement.
    from ..model import Confidence, Edge, GraphStore, Node

    store = GraphStore()
    store.add_node(Node(id="fn:same.py::a", kind="FUNCTION", attrs={}, path="same.py", line=1))
    store.add_node(Node(id="lit:p", kind="LITERAL", attrs={"raw": "x.foo", "normalized": "x.foo", "flavour": "path", "tail": "x.foo"}))
    store.add_node(Node(id="lit:c", kind="LITERAL", attrs={"raw": "x.bar", "normalized": "x.bar", "flavour": "path", "tail": "x.bar"}))
    store.add_edge(Edge(src="fn:same.py::a", dst="lit:p", kind="PRODUCES_LITERAL", confidence=Confidence.PROVEN,
                         attrs={"role": "test", "line": 1, "col": 0}))
    store.add_edge(Edge(src="fn:same.py::a", dst="lit:c", kind="CONSUMES_LITERAL", confidence=Confidence.PROVEN,
                         attrs={"role": "test", "line": 2, "col": 0}))
    producer_only, consumer_only = orphans(store, flavour="path")
    pairs = near_miss_pairs(producer_only, consumer_only)
    assert pairs == []


def test_materialize_near_miss_edges_adds_edges_to_store():
    store, _, _ = _build()
    producer_only, consumer_only = orphans(store, flavour="path")
    pairs = near_miss_pairs(producer_only, consumer_only)
    before = len(list(store.edges_of_kind("NEAR_MISS")))
    edges = materialize_near_miss_edges(store, pairs)
    after = len(list(store.edges_of_kind("NEAR_MISS")))
    assert after - before == len(edges)
    assert len(edges) >= 1
    for e in edges:
        assert e.confidence.name == "PROBABLE"
