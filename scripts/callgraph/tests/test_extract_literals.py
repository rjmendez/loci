"""extract/literals.py: LITERAL/PATHEXPR nodes and PRODUCES_LITERAL /
CONSUMES_LITERAL edges, against real GraphStore output over fixtures."""
from ..extract.literals import (
    classify_flavour, literal_node_id, looks_key_like, looks_path_like, normalize_literal, tail_segment,
)
from ..tests.helpers import build_fixture_store


def test_looks_path_like():
    assert looks_path_like("~/.hermes/**/graph.kuzu")
    assert looks_path_like("mcp/server.py")
    assert looks_path_like("graph.ladybug")
    assert not looks_path_like("LOCI_PORT")
    assert not looks_path_like("")


def test_looks_key_like():
    assert looks_key_like("LOCI_PORT")
    assert looks_key_like("cfg_lookup_only")
    assert not looks_key_like("~/.hermes/**/graph.kuzu")
    assert not looks_key_like("has a space")


def test_normalize_collapses_double_star():
    assert normalize_literal("~/.hermes/**/graph.kuzu") == "~/.hermes/*/graph.kuzu"


def test_tail_segment_last_path_component():
    assert tail_segment("~/.hermes/*/graph.kuzu") == "graph.kuzu"
    assert tail_segment("graph.ladybug") == "graph.ladybug"


def test_classify_flavour():
    assert classify_flavour("mcp/server.py") == "path"
    assert classify_flavour("LOCI_PORT") == "key"
    assert classify_flavour("has a space and/slash") is None or classify_flavour("has a space and/slash") == "path"


def test_literal_node_id_stable_for_same_normalized_text():
    a = literal_node_id(normalize_literal("~/.hermes/**/graph.kuzu"))
    b = literal_node_id(normalize_literal("~/.hermes/*/graph.kuzu"))
    assert a == b


# -- store-ctor PATHEXPR producer (BUG D producer shape) ---------------------


def test_store_ctor_pathexpr_produces_tail_literal():
    store, _, _ = build_fixture_store(["literals_produce.py"])
    pathexprs = [n for n in store.nodes_of_kind("PATHEXPR") if n.attrs.get("tail_literal") == "graph.ladybug"]
    assert len(pathexprs) == 1
    pid = pathexprs[0].id
    edges = store.in_edges(pid, "PRODUCES_LITERAL")
    assert len(edges) == 1
    assert edges[0].attrs["role"] == "store-ctor"
    assert edges[0].src == "fn:literals_produce.py::open_store"


def test_write_text_on_composed_path_produces():
    store, _, _ = build_fixture_store(["literals_produce.py"])
    pathexprs = [n for n in store.nodes_of_kind("PATHEXPR") if n.attrs.get("tail_literal") == "notes.jsonl"]
    assert len(pathexprs) == 1
    edges = store.in_edges(pathexprs[0].id, "PRODUCES_LITERAL")
    assert any(e.attrs["role"] == "write_text" for e in edges)


# -- glob/open consumer shapes (BUG D consumer shape) -------------------------


def test_glob_with_expanduser_wrapper_consumes_bare_literal():
    store, _, _ = build_fixture_store(["literals_consume.py", "literals_produce.py"])
    normalized = normalize_literal("~/.hermes/**/graph.kuzu")
    lid = literal_node_id(normalized)
    node = store.get(lid)
    assert node is not None
    assert node.kind == "LITERAL"
    assert node.attrs["flavour"] == "path"
    edges = store.in_edges(lid, "CONSUMES_LITERAL")
    assert len(edges) == 1
    assert edges[0].attrs["role"] == "glob"
    assert edges[0].src == "fn:literals_consume.py::find_databases"


def test_open_default_mode_consumes_composed_path():
    store, _, _ = build_fixture_store(["literals_consume.py", "literals_produce.py"])
    pathexprs = [n for n in store.nodes_of_kind("PATHEXPR") if n.attrs.get("tail_literal") == "notes.jsonl"]
    consume_edges = [e for pe in pathexprs for e in store.in_edges(pe.id, "CONSUMES_LITERAL")]
    assert any(e.attrs["role"] == "open-r" and e.src == "fn:literals_consume.py::read_note" for e in consume_edges)


# -- key-like: environ + dict get/subscript -----------------------------------


def test_environ_get_and_subscript_share_one_literal_both_directions():
    store, _, _ = build_fixture_store(["literals_keys.py"])
    lid = literal_node_id(normalize_literal("LOCI_PORT"))
    node = store.get(lid)
    assert node is not None and node.attrs["flavour"] == "key"
    consumes = store.in_edges(lid, "CONSUMES_LITERAL")
    produces = store.in_edges(lid, "PRODUCES_LITERAL")
    assert any(e.attrs["role"] == "environ-get" for e in consumes)
    assert any(e.attrs["role"] == "environ-set" for e in produces)


def test_dict_get_and_subscript_orphan_keys_are_distinct():
    store, _, _ = build_fixture_store(["literals_keys.py"])
    lookup_id = literal_node_id(normalize_literal("cfg_lookup_only"))
    write_id = literal_node_id(normalize_literal("cfg_write_only"))
    assert store.in_edges(lookup_id, "CONSUMES_LITERAL")
    assert not store.in_edges(lookup_id, "PRODUCES_LITERAL")
    assert store.in_edges(write_id, "PRODUCES_LITERAL")
    assert not store.in_edges(write_id, "CONSUMES_LITERAL")
