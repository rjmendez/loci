"""extract/flow.py: escaping-local detection (LOCALBINDING/ESCAPES) and the
per-function control-flow summary the flag audit reads — BUG B's shape, in
miniature, against real GraphStore output over fixtures/flags_shapes.py."""
from ..extract.flow import local_binding_id
from ..tests.helpers import build_fixture_store

FN = "fn:flags_shapes.py::assemble"
OK = local_binding_id(FN, "ok")


def _store():
    store, _, _ = build_fixture_store(["flags_shapes.py"])
    return store


def test_ok_is_constant_initialized_and_dict_value_escape():
    store = _store()
    node = store.get(OK)
    assert node is not None
    assert node.attrs["init_is_constant"] is True
    edges = [e for e in store.out_edges(FN, "ESCAPES") if e.dst == OK]
    assert len(edges) == 1
    assert edges[0].attrs["escape_form"] == "dict-value"


def test_ok_has_two_guarded_reassignments_tagged_if_branch():
    store = _store()
    node = store.get(OK)
    contexts = [c["context"] for c in node.attrs["assign_contexts"]]
    assert contexts == ["if-branch", "if-branch"]
    assert len(node.attrs["assign_lines"]) == 2


def test_ok_has_three_guard_exits_none_of_which_assign_it():
    store = _store()
    node = store.get(OK)
    guard_exits = node.attrs["guard_exits"]
    assert len(guard_exits) == 3
    kinds = sorted(g["kind"] for g in guard_exits)
    assert kinds == ["break", "continue", "continue"]


def test_parts_escapes_but_is_not_constant_initialized():
    store = _store()
    parts_id = local_binding_id(FN, "parts")
    node = store.get(parts_id)
    assert node is not None
    assert node.attrs["init_is_constant"] is False
    edges = [e for e in store.out_edges(FN, "ESCAPES") if e.dst == parts_id]
    assert edges[0].attrs["escape_form"] == "dict-value"


def test_direct_return_escape_form():
    fn = "fn:flags_shapes.py::direct_return_case"
    store = _store()
    lid = local_binding_id(fn, "result")
    node = store.get(lid)
    assert node is not None
    edges = [e for e in store.out_edges(fn, "ESCAPES") if e.dst == lid]
    assert edges[0].attrs["escape_form"] == "direct-return"


def test_tuple_element_escape_form_and_params_excluded():
    fn = "fn:flags_shapes.py::tuple_return_case"
    store = _store()
    total_id = local_binding_id(fn, "total")
    node = store.get(total_id)
    assert node is not None
    edges = [e for e in store.out_edges(fn, "ESCAPES") if e.dst == total_id]
    assert edges[0].attrs["escape_form"] == "tuple-element"
    # `a` and `b` are parameters returned as-is, not locally (re)bound —
    # must NOT get their own LOCALBINDING (see module docstring).
    assert store.get(local_binding_id(fn, "a")) is None
    assert store.get(local_binding_id(fn, "b")) is None


def test_closure_escape_marks_free_variables_of_returned_nested_function():
    fn = "fn:flags_shapes.py::make_adder"
    store = _store()
    base_id = local_binding_id(fn, "base")
    step_id = local_binding_id(fn, "step")
    base_node = store.get(base_id)
    step_node = store.get(step_id)
    assert base_node is not None and step_node is not None
    base_edges = [e for e in store.out_edges(fn, "ESCAPES") if e.dst == base_id]
    step_edges = [e for e in store.out_edges(fn, "ESCAPES") if e.dst == step_id]
    assert base_edges[0].attrs["escape_form"] == "closure"
    assert step_edges[0].attrs["escape_form"] == "closure"
    assert step_node.attrs["init_is_constant"] is True   # `step = 1`
    assert base_node.attrs["init_is_constant"] is False  # `base = n`
    # the nested function's OWN name, returned by itself, is not a "local"
    # of make_adder in the sense this module models.
    assert store.get(local_binding_id(fn, "add")) is None


def test_no_escape_case_creates_no_localbinding():
    fn = "fn:flags_shapes.py::no_escape_case"
    store = _store()
    lid = local_binding_id(fn, "unused_local")
    assert store.get(lid) is None
    assert store.out_edges(fn, "ESCAPES") == []
