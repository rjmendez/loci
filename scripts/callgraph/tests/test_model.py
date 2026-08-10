"""model.py: GraphStore's (de)serialization and DOT export, against a
small hand-built store -- not the real corpus, so these stay fast and the
expected output is something a human can read in the assertion itself."""
from ..model import Confidence, Edge, GraphStore, Node


def _store():
    store = GraphStore()
    store.add_node(Node("fn:a.py::foo", "FUNCTION", {"qualname": "foo"}, path="a.py", line=1))
    store.add_node(Node("fn:b.py::bar", "FUNCTION", {"qualname": "bar"}, path="b.py", line=1))
    store.add_node(Node("mod:a.py", "MODULE", {}, path="a.py", line=1))
    store.add_edge(Edge("mod:a.py", "fn:a.py::foo", "DEFINES", Confidence.PROVEN))
    store.add_edge(Edge("call:a.py:3:0", "fn:b.py::bar", "CALLS", Confidence.PROBABLE, {"rung": "x"}))
    return store


def test_json_round_trip_preserves_nodes_edges_and_confidence():
    store = _store()
    text = store.to_json()
    restored = GraphStore.from_json(text)
    assert set(restored.nodes) == set(store.nodes)
    assert len(restored.edges) == len(store.edges)
    call_edges = list(restored.edges_of_kind("CALLS"))
    assert len(call_edges) == 1
    assert call_edges[0].confidence == Confidence.PROBABLE
    assert call_edges[0].attrs["rung"] == "x"


def test_stats_counts_by_kind():
    store = _store()
    stats = store.stats()
    assert stats["node:FUNCTION"] == 2
    assert stats["node:MODULE"] == 1
    assert stats["edge:DEFINES"] == 1
    assert stats["edge:CALLS"] == 1
    assert stats["nodes"] == 3
    assert stats["edges"] == 2


def test_to_dot_whole_graph_includes_every_node_and_edge():
    store = _store()
    dot = store.to_dot()
    assert dot.startswith("digraph callgraph {")
    assert dot.rstrip().endswith("}")
    assert '"fn:a.py::foo"' in dot
    assert '"fn:b.py::bar"' in dot
    assert '"mod:a.py" -> "fn:a.py::foo"' in dot
    assert 'label="foo\\n[FUNCTION]"' in dot


def test_to_dot_confidence_maps_to_line_style():
    store = _store()
    dot = store.to_dot()
    # DEFINES is PROVEN -> solid; the CALLS edge is PROBABLE -> dashed.
    lines = {l for l in dot.splitlines() if "->" in l}
    defines_line = next(l for l in lines if "DEFINES" in l)
    calls_line = next(l for l in lines if "CALLS" in l)
    assert "style=solid" in defines_line
    assert "style=dashed" in calls_line


def test_to_dot_filtered_subgraph_excludes_unselected_nodes():
    store = _store()
    dot = store.to_dot(node_ids={"fn:a.py::foo", "mod:a.py"}, edges=[])
    assert '"fn:a.py::foo"' in dot
    assert '"fn:b.py::bar"' not in dot
    assert "->" not in dot  # edges=[] means no edge lines at all


def test_to_dot_escapes_quotes_in_labels():
    store = GraphStore()
    store.add_node(Node('fn:a.py::weird"name', "FUNCTION", {"qualname": 'weird"name'}, path="a.py", line=1))
    dot = store.to_dot()
    assert '\\"' in dot
    # the raw (unescaped) quote must never appear inside a quoted DOT string
    assert 'weird"name"' not in dot
