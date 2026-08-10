"""analyze/flags.py: the partial-assignment ranker — BUG B's shape, in
miniature, against real GraphStore output over fixtures/flags_shapes.py."""
from ..analyze.flags import rank_flags
from ..tests.helpers import build_fixture_store


def _store():
    store, _, _ = build_fixture_store(["flags_shapes.py"])
    return store


def test_rank_flags_only_includes_constant_initialized_escaping_locals():
    store = _store()
    rows = rank_flags(store)
    idents = {r.ident for r in rows}
    # `ok` (constant False init) and `step` (constant 1 init, closure
    # escape) both qualify; `parts`/`base`/`result`/`total` do not (their
    # inits are not ast.Constant).
    assert "ok" in idents
    assert "step" in idents
    assert "parts" not in idents
    assert "base" not in idents
    assert "result" not in idents
    assert "total" not in idents


def test_ok_ranks_above_step_by_guard_exit_ratio():
    store = _store()
    rows = rank_flags(store)
    by_ident = {r.ident: r for r in rows}
    assert by_ident["ok"].score > by_ident["step"].score
    assert rows[0].ident == "ok"          # sorted descending by score
    assert by_ident["ok"].score == 3 / 2  # 3 guard exits / 2 guarded assigns
    assert by_ident["step"].score == 0.0  # never reassigned, no guard exits either


def test_fn_filter_scopes_to_single_function():
    store = _store()
    rows = rank_flags(store, fn_filter="fn:flags_shapes.py::assemble")
    assert {r.ident for r in rows} == {"ok"}


def test_scope_prefix_filter():
    store = _store()
    rows = rank_flags(store, scope_prefixes=["nonexistent/"])
    assert rows == []
