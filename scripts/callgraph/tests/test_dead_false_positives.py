"""FALSE-POSITIVE GATE for `cg dead`.

A dead-code query is killed by its false positives, not by its misses. If it
names functions that are obviously live, people stop reading it, and a tool
people suppress is a tool that is not working.

Two layers here:

1. THE HARD GATE, over the real corpus at HEAD: zero registered function may
   ever be reported dead. This is an absolute, not a ratio.

2. One test per DISPATCH SHAPE that produced a false positive during
   validation, each against a minimal fixture. These are the shapes that took
   `cg dead` from 165 rows to 66 on this corpus. They are separated from the
   corpus test on purpose: the corpus number will drift as the code changes,
   but "a function passed by bare name is not dead" must hold forever.

The remaining 66 rows are NOT all false positives and are NOT asserted to
zero — see docs/LIMITS.md for the hand-verified breakdown. 53 of them are
methods reached only through an attribute call on an untyped receiver, which
is a genuine limit of a tool with no type inference, not a bug to be fixed by
loosening this file.
"""
from __future__ import annotations


from ..analyze.deadcode import dead_functions, registered_but_dead
from .helpers import build_fixture_store


# ---------------------------------------------------------------------------
# 1. the hard gate, on the real corpus
# ---------------------------------------------------------------------------


def test_no_registered_function_is_ever_reported_dead(head_build):
    """40 @mcp.tool(), 31 register()d, 13 _SKILL_MAP, 6 FastAPI routes,
    1 resource, 1 custom_route = 92 registered functions. None may be dead."""
    store = head_build.store
    registered = {e.dst for e in store.edges_of_kind("REGISTERS")}
    assert len(registered) == 92, f"registration surface changed: {len(registered)}"

    offenders = registered_but_dead(store)
    assert offenders == [], (
        f"{len(offenders)} registered function(s) reported dead: "
        f"{[n.id for n in offenders]}"
    )


def test_mcp_tool_and_manifest_surfaces_specifically(head_build):
    """The two shapes named in the brief, asserted by name rather than folded
    into the aggregate above, so a failure says WHICH surface broke."""
    store = head_build.store
    dead_ids = {n.id for n in dead_functions(store)}

    tools = {e.dst for e in store.edges_of_kind("REGISTERS")
             if e.src == "reg:mcp/server.py::mcp.tool"}
    manifest = {e.dst for e in store.edges_of_kind("REGISTERS")
                if (src := store.get(e.src)) is not None
                and src.attrs.get("mechanism") == "manifest-tuple"}

    assert len(tools) == 40, f"@mcp.tool() count changed: {len(tools)}"
    assert len(manifest) == 31, f"register() manifest count changed: {len(manifest)}"
    assert tools & dead_ids == set()
    assert manifest & dead_ids == set()


def test_dead_row_count_does_not_regress(head_build):
    """A ceiling, not a target. 66 rows at the HEAD this was validated against
    (down from 165 before the five resolution fixes + the dunder filter). If this climbs sharply,
    a resolution path has broken and the query is filling up with noise again.
    """
    n = len(dead_functions(head_build.store))
    assert n <= 85, (
        f"`cg dead` reports {n} unreachable functions (was 66 when validated). "
        "A jump means a dispatch shape stopped resolving — find it before "
        "raising this number."
    )


# ---------------------------------------------------------------------------
# 2. one test per false-positive shape found during validation
# ---------------------------------------------------------------------------


def _dead_names(*sources) -> set[str]:
    store, _, _ = build_fixture_store(list(sources))
    return {n.attrs["qualname"] for n in dead_functions(store)}


def test_bare_name_passed_as_call_argument_is_not_dead():
    """mcp/server.py:
        checks.append(_health_check("embeddings_sparse", _health_probe_embeddings_sparse))
    Nine `_health_probe_*` functions were reported dead this way."""
    assert "probe_sparse" not in _dead_names("regress/fp_shapes/callback_arg.py")


def test_function_called_only_inside_a_lambda_body_is_not_dead():
    """mcp/server.py:
        ("qdrant_reachable", lambda: backends.qdrant()[0]),
    A lambda is consumed where it is written, so its body is reachable exactly
    when the scope constructing it is."""
    assert "probe_via_lambda" not in _dead_names("regress/fp_shapes/lambda_call.py")


def test_function_local_import_is_visible_inside_a_nested_scope():
    """`import backends` in loci_health, used as `backends.qdrant()` from
    inside a lambda nested in it. Python closures make the binding visible;
    resolution has to walk outward through the <locals> chain."""
    names = _dead_names("regress/fp_shapes/lazy_import_nested.py",
                        "regress/fp_shapes/lazy_import_target.py")
    assert "helper" not in names


def test_sibling_nested_def_called_by_bare_name_is_not_dead():
    """mcp/graph/code_parse.py's parse_source defines enclosing_def_node and
    _in_method side by side; _in_method calls enclosing_def_node(node)."""
    names = _dead_names("regress/fp_shapes/nested_siblings.py")
    assert "outer.<locals>.sibling_a" not in names


def test_protocol_dunders_are_not_reported(head_build):
    """`with _Mutex(...)` invokes __enter__/__exit__ without ever naming them.
    Asserted on the real corpus because that is where the shape occurred."""
    names = {n.attrs["qualname"] for n in dead_functions(head_build.store)}
    assert "_Mutex.__enter__" not in names
    assert "_Mutex.__exit__" not in names
    assert not any(q.rsplit(".", 1)[-1].startswith("__") for q in names)


def test_relative_from_import_of_a_submodule_resolves():
    """mcp/graph/analytics.py: `from . import queries as Q` then
    `Q.finding_symbols(...)`. Four live queries.py functions were reported
    dead because Q aliased an external sink instead of the MODULE."""
    names = _dead_names("regress/fp_shapes/pkg/__init__.py",
                        "regress/fp_shapes/pkg/queries.py",
                        "regress/fp_shapes/pkg/analytics.py")
    assert "finding_symbols" not in names


# ---------------------------------------------------------------------------
# 3. the counter-test: the fixes must not make EVERYTHING reachable
# ---------------------------------------------------------------------------


def test_genuinely_unreferenced_function_is_still_reported():
    """The above fixes are deliberately broad. If they were broad enough to
    swallow a function that nothing mentions at all, `cg dead` would be
    worthless — it would simply never report anything."""
    names = _dead_names("regress/fp_shapes/truly_dead.py")
    assert "never_mentioned_anywhere" in names
    assert "used" not in names          # called by main()
    assert "passed_around" not in names  # bare-name escape only
