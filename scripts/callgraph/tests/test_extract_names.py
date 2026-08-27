"""extract/names.py: WRITES_NAME (module-level-assign | global-stmt |
import-binding | def-binding) and READS_NAME (same-module and cross-module),
against real GraphStore output."""
from ..model import Confidence
from ..tests.helpers import build_fixture_store


def test_module_level_assign_write():
    store, _, _ = build_fixture_store(["dangling_global.py"])
    nid = "name:dangling_global.py::_real_counter"
    writes = store.in_edges(nid, "WRITES_NAME")
    kinds = {(w.src, w.attrs["via"]) for w in writes}
    assert ("mod:dangling_global.py", "module-level-assign") in kinds


def test_global_stmt_write_line_and_src():
    store, _, _ = build_fixture_store(["dangling_global.py"])
    nid = "name:dangling_global.py::_symbol_index_count"
    writes = store.in_edges(nid, "WRITES_NAME")
    assert len(writes) == 1
    w = writes[0]
    assert w.src == "fn:dangling_global.py::invalidate_cache"
    assert w.attrs["via"] == "global-stmt"
    assert w.attrs["line"] == 11  # `_symbol_index_count = 0` inside invalidate_cache


def test_def_binding_write():
    store, _, _ = build_fixture_store(["dangling_global.py"])
    nid = "name:dangling_global.py::invalidate_cache"
    writes = store.in_edges(nid, "WRITES_NAME")
    assert any(w.attrs["via"] == "def-binding" and w.src == "mod:dangling_global.py" for w in writes)


def test_augmented_write_tagged_separately_from_plain_assign():
    store, _, _ = build_fixture_store(["dangling_global.py"])
    # scopes.py lumps assign+augassign; WRITES_NAME must still emit a separate edge.
    nid = "name:dangling_global.py::_real_counter"
    writes = store.in_edges(nid, "WRITES_NAME")
    via_by_src = {w.src: w.attrs["via"] for w in writes}
    assert via_by_src["fn:dangling_global.py::bump"] == "global-stmt"
    assert via_by_src["mod:dangling_global.py"] == "module-level-assign"


def test_reads_name_in_call_position():
    store, _, _ = build_fixture_store(["calls_shapes.py", "calls_target.py"])
    nid = "name:calls_shapes.py::helper"
    reads = store.in_edges(nid, "READS_NAME")
    assert len(reads) == 1
    assert reads[0].src == "fn:calls_shapes.py::caller_name_def_local"
    assert reads[0].attrs["in_call_position"] is True


def test_local_variable_is_not_misreported_as_a_global_read():
    # A parameter is a genuine local: no NAME node, hence no spurious READS_NAME.
    store, _, _ = build_fixture_store(["calls_shapes.py", "calls_target.py"])
    assert store.get("name:calls_shapes.py::callback") is None


def test_cross_module_reads_name_via_imported_module_attribute():
    store, _, _ = build_fixture_store(["names_cross_module.py", "names_cross_module_target.py"])
    nid = "name:names_cross_module_target.py::CONFIG_VALUE"
    reads = store.in_edges(nid, "READS_NAME")
    assert len(reads) == 1
    r = reads[0]
    assert r.src == "fn:names_cross_module.py::read_it"
    assert r.attrs["cross_module"] is True
    assert r.confidence == Confidence.PROVEN
