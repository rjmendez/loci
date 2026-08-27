"""analyze/nameaudit.py: DANGLING-GLOBAL (BUG C) and WRITE-WITH-NO-READ,
against real GraphStore output over fixtures."""
from ..analyze.nameaudit import dangling_globals, read_by_tests, write_no_read
from ..tests.helpers import build_fixture_store


def test_dangling_global_flags_global_only_slot():
    store, _, _ = build_fixture_store(["dangling_global.py"])
    findings = dangling_globals(store)
    slots = {f.slot.id for f in findings}
    assert "name:dangling_global.py::_symbol_index_cache" in slots
    assert "name:dangling_global.py::_symbol_index_count" in slots


def test_dangling_global_excludes_module_level_bound_ident():
    # Module-level-bound beats a `global` redeclaration: never dangling.
    store, _, _ = build_fixture_store(["dangling_global.py"])
    findings = dangling_globals(store)
    slots = {f.slot.id for f in findings}
    assert "name:dangling_global.py::_real_counter" not in slots


def test_dangling_global_names_the_real_slot_in_another_module():
    store, _, _ = build_fixture_store(["dangling_global.py", "dangling_global_real_slot.py"])
    findings = {f.slot.id: f for f in dangling_globals(store)}
    f = findings["name:dangling_global.py::_symbol_index_cache"]
    assert len(f.real_slots) == 1
    real = f.real_slots[0]
    assert real.path == "dangling_global_real_slot.py"
    assert real.line == 7
    assert real.id == "name:dangling_global_real_slot.py::_symbol_index_cache"


def test_dangling_global_write_sites_recorded():
    store, _, _ = build_fixture_store(["dangling_global.py"])
    findings = {f.slot.id: f for f in dangling_globals(store)}
    f = findings["name:dangling_global.py::_symbol_index_count"]
    assert len(f.global_writes) == 1
    assert f.global_writes[0].src == "fn:dangling_global.py::invalidate_cache"


def test_dangling_global_no_real_slot_found_leaves_list_empty():
    # Without the real-slot fixture nothing in the corpus binds `_symbol_index_cache`.
    store, _, _ = build_fixture_store(["dangling_global.py"])
    findings = {f.slot.id: f for f in dangling_globals(store)}
    assert findings["name:dangling_global.py::_symbol_index_cache"].real_slots == []


def test_dangling_global_scope_filter():
    store, _, _ = build_fixture_store(["dangling_global.py", "dangling_global_real_slot.py"])
    findings = dangling_globals(store, scope_prefixes=["dangling_global_real_slot.py"])
    assert findings == []


def test_write_no_read_flags_write_only_slot():
    # `_INTERNAL_STATE` is module-level-bound and read by nothing in this fixture set.
    store, _, _ = build_fixture_store(["reexport.py", "reexport_source.py"])
    findings = {f.slot.id for f in write_no_read(store)}
    assert "name:reexport_source.py::_INTERNAL_STATE" in findings


def test_write_no_read_excludes_slot_with_a_read():
    # calls_shapes.py::helper has both a write (def-binding) and a read: never write-only.
    store, _, _ = build_fixture_store(["calls_shapes.py", "calls_target.py"])
    findings = {f.slot.id for f in write_no_read(store)}
    assert "name:calls_shapes.py::helper" not in findings


def test_write_no_read_scope_filter():
    store, _, _ = build_fixture_store(["reexport.py", "reexport_source.py"])
    findings = write_no_read(store, scope_prefixes=["reexport.py"])
    assert all(f.slot.path == "reexport.py" for f in findings)
    assert not any(f.slot.id == "name:reexport_source.py::_INTERNAL_STATE" for f in findings)


def test_read_by_tests_matches_whole_word_token():
    from ..ingest import SourceFile

    test_src = SourceFile(
        rel_path="mcp/tests/test_thing.py",
        source="from reexport_source import _INTERNAL_STATE\n\ndef test_x():\n    assert _INTERNAL_STATE is None\n",
        tree=None, error=None, origin="fixture", sha1="",
    )
    hits = read_by_tests({"_INTERNAL_STATE", "totally_unused_ident"}, [test_src])
    assert hits == {"_INTERNAL_STATE": ["mcp/tests/test_thing.py"]}
