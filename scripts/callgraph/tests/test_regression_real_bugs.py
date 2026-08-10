"""REGRESSION GATE: the three structurally-visible bugs this tool exists to catch.

Every other test in this package runs against fixtures that were WRITTEN to
demonstrate a shape. These run against the REAL pre-fix code, copied verbatim
out of git history into fixtures/regress/ (see that directory's generator
provenance headers). The distinction matters: a hand-written fixture proves the
analyzer handles a shape its own author had in mind, and nothing more. These
prove it handles the code that actually shipped the bug -- 316 lines of
grounding.py with ten lanes of unrelated control flow, 408 lines of
graph_tools.py, the real glob call in graph_facts.py.

If a later change blinds the tool, these fail. That is the entire point:
`selftest`'s "BUG x shape" checks can keep passing against the toy fixtures
long after the analyzer has stopped working on real input.

BUG A (mcp/graph/queries.py's node["_id"] vs the backend's "_ID") is
DELIBERATELY ABSENT. It is a data-format mismatch between this code and a
value produced at runtime by another library. Nothing in the source text
distinguishes the correct key from the wrong one -- both are string
subscripts of an opaque dict -- so no call graph can see it, and pretending
otherwise by special-casing key casing would be a contrived check that
earns false confidence. See docs/LIMITS.md.
"""
from __future__ import annotations

import subprocess

import pytest

from ..analyze.flags import rank_flags
from ..analyze.literalaudit import near_miss_pairs, orphans
from ..analyze.nameaudit import dangling_globals
from ..config import REPO_ROOT
from .helpers import FIXTURES_DIR, build_fixture_store

MARKER = "# ---- VERBATIM BODY BELOW (do not edit; see test_regression_real_bugs.py) ----\n"


# ---------------------------------------------------------------------------
# fixture provenance -- the fixtures must stay VERBATIM copies of real history
# ---------------------------------------------------------------------------

# (fixture rel path, commit-ish, source path in that commit, line slices or None)
PROVENANCE = [
    ("regress/bug_b/grounding.py", "d359e9a", "mcp/grounding.py", None),
    ("regress/bug_c/graph_tools.py", "c1c40a9^", "mcp/graph_tools.py", None),
    ("regress/bug_c/ladybug_ops.py", "c1c40a9^", "mcp/ladybug_ops.py", None),
    ("regress/bug_d/graph_facts.py", "08c9198^", "scripts/graph_facts.py", None),
    ("regress/bug_d/server.py", "08c9198^", "mcp/server.py", [(159, 166), (218, 287)]),
]


def _git_show(rev: str, path: str) -> str | None:
    try:
        p = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


@pytest.mark.parametrize("rel,rev,src,slices", PROVENANCE, ids=[p[0] for p in PROVENANCE])
def test_fixture_is_verbatim_pre_fix_code(rel, rev, src, slices):
    """The fixture body must still be byte-identical to what git says shipped.

    Without this, a future maintainer 'fixing' a failing regression test by
    editing the fixture would silently turn it back into a toy -- the exact
    failure mode this whole file exists to prevent.
    """
    real = _git_show(rev, src)
    if real is None:
        pytest.skip(f"git history for {rev}:{src} unavailable")
    body = (FIXTURES_DIR / rel).read_text().split(MARKER, 1)[1]
    if slices is None:
        assert body == real, f"{rel} has drifted from {rev}:{src}"
    else:
        lines = real.splitlines(keepends=True)
        for a, b in slices:
            chunk = "".join(lines[a - 1:b])
            assert chunk in body, f"{rel} no longer contains {src} lines {a}-{b} verbatim"


# ---------------------------------------------------------------------------
# BUG B -- degraded assigned on only some control-flow paths before return
# ---------------------------------------------------------------------------


def test_bug_b_flags_ranks_degraded_in_real_grounding():
    """`cg flags mcp/grounding.py` must surface `degraded`.

    Real numbers on the real pre-fix file: init `False` at the top of
    ground(), 3 guarded reassignments to True, 11 guarded exits (break/
    continue in `if not S:` guards and except handlers) that skip it
    entirely -> score 11/3 = 3.67, and it is the ONLY candidate in the
    whole module.
    """
    store, _, _ = build_fixture_store(["regress/bug_b/grounding.py"])
    rows = rank_flags(store)

    assert [r.ident for r in rows] == ["degraded"], (
        "degraded must be the sole flag candidate in the real grounding.py; got "
        f"{[(r.ident, r.fn_id) for r in rows]}"
    )
    row = rows[0]
    assert row.fn_id.endswith("::ground")
    assert row.escape_form == "dict-value"        # returned inside the result dict
    assert row.binding.attrs["init_is_constant"] is True
    # The defect IS this asymmetry: far more paths skip the flag than set it.
    assert len(row.binding.attrs["guard_exits"]) == 11
    assert len(row.binding.attrs["constant_rebind_lines"]) == 3
    assert row.score == pytest.approx(11 / 3)
    assert row.pattern == "flag"


def test_bug_b_survives_whole_corpus_ranking():
    """Scoped to the file the tool is decisive; over the WHOLE corpus it is a
    ranker and `degraded` is merely near the top, not first.

    Pinned deliberately: measured at rev d359e9a (where the bug was live)
    `degraded` is 3rd of 7 rows. The two above it -- canary.py's
    `rollback_recommended` and server.py's `sampling_mode` -- are the same
    shape in CORRECT code, which static analysis cannot distinguish from a
    real one. If a future change pushes `degraded` far down this list, the
    ranker has stopped working even though the scoped test still passes.
    """
    pytest.importorskip("subprocess")
    from ..pipeline import build_graph
    if _git_show("d359e9a", "mcp/grounding.py") is None:
        pytest.skip("git history unavailable")
    res = build_graph(rev="d359e9a")
    rows = rank_flags(res.store)
    idx = [i for i, r in enumerate(rows)
           if r.ident == "degraded" and (r.binding.path or "").endswith("grounding.py")]
    assert idx, "degraded vanished from the whole-corpus flag ranking"
    assert idx[0] < 5, f"degraded fell to rank {idx[0] + 1} of {len(rows)}"
    assert len(rows) <= 12, (
        f"flag ranker got noisier: {len(rows)} rows corpus-wide (was 7). "
        "A row count that climbs is how this query becomes unreadable."
    )


def test_accumulators_are_not_ranked_as_flags():
    """The false-positive class that made BUG B rank 7th of 23 before this
    validation pass: counters. `created = 0 ... created += len(chunk)` matches
    "constant init, updated on only some paths" exactly, and there were 16 of
    them drowning the real bug."""
    store, _, _ = build_fixture_store(["regress/accumulator_fp.py"])

    default_rows = {r.ident for r in rank_flags(store)}
    assert "n_drifted" not in default_rows      # pure `+=` counter
    assert "lines_scanned" not in default_rows  # hybrid: reset to a constant, then `+=`
    assert "degraded" in default_rows           # the genuine flag in the same file

    with_acc = {r.ident: r for r in rank_flags(store, include_accumulators=True)}
    assert with_acc["n_drifted"].pattern == "accumulator"
    assert with_acc["lines_scanned"].pattern == "accumulator"
    assert with_acc["degraded"].pattern == "flag"


# ---------------------------------------------------------------------------
# BUG C -- module-level name written by exactly one module and read by none
# ---------------------------------------------------------------------------


def test_bug_c_dangling_global_in_real_graph_tools():
    """`cg writes-dead` must report both phantom globals AND name the real slot.

    Naming the real slot is what makes this a diagnosis rather than a lint
    warning: it is the difference between "this global is odd" and "the cache
    you meant to invalidate lives in ladybug_ops".
    """
    store, _, _ = build_fixture_store([
        "regress/bug_c/graph_tools.py",
        "regress/bug_c/ladybug_ops.py",
    ])
    findings = dangling_globals(store)
    by_ident = {f.slot.id.rsplit("::", 1)[-1]: f for f in findings}

    assert set(by_ident) == {"_symbol_index_cache", "_symbol_index_count"}, (
        f"expected exactly the two phantom globals, got {sorted(by_ident)}"
    )
    for ident, f in by_ident.items():
        assert f.slot.path == "regress/bug_c/graph_tools.py"
        assert "global-only" in set(f.slot.attrs["binding_kind"])
        # the write is attributed to the function that actually made it
        assert [e.src.rsplit("::", 1)[-1] for e in f.global_writes] == ["code_memory_relink"]
        assert all(e.attrs["via"] == "global-stmt" for e in f.global_writes)
        # and the REAL slot is found, in the other module, module-level bound
        assert [n.path for n in f.real_slots] == ["regress/bug_c/ladybug_ops.py"], (
            f"{ident}: real slot not located in ladybug_ops"
        )


def test_bug_c_would_not_fire_once_fixed():
    """Mutation check: the SAME analyzer over the FIXED shape must stay quiet.

    A detector that fires on the fix as well as the bug is not a detector.
    """
    store, _, _ = build_fixture_store(["regress/bug_c_fixed.py"])
    assert dangling_globals(store) == []


# ---------------------------------------------------------------------------
# BUG D -- a path literal produced in one module, near-missed in another
# ---------------------------------------------------------------------------


def test_bug_d_literal_orphans_in_real_code():
    """`cg literals --paths --orphans` must show graph.ladybug produced and
    never consumed, and graph.kuzu consumed and never produced."""
    store, _, _ = build_fixture_store([
        "regress/bug_d/server.py",
        "regress/bug_d/graph_facts.py",
    ])
    producer_only, consumer_only = orphans(store, flavour="path")

    prod = {g.match_text: g for g in producer_only}
    cons = {g.match_text: g for g in consumer_only}
    assert "graph.ladybug" in prod, f"producer orphans were {sorted(prod)}"
    assert "graph.kuzu" in cons, f"consumer orphans were {sorted(cons)}"

    # attributed to the right side of the split
    assert {s.path for s in prod["graph.ladybug"].produce_sites} == {"regress/bug_d/server.py"}
    assert {s.path for s in cons["graph.kuzu"].consume_sites} == {"regress/bug_d/graph_facts.py"}
    assert prod["graph.ladybug"].produce_sites[0].role == "store-ctor"
    assert cons["graph.kuzu"].consume_sites[0].role == "glob"


def test_bug_d_near_miss_pairs_the_two_halves():
    """The orphan pair is the lead; --near-miss is what puts the two halves on
    one line, across the module boundary, on the shared `graph` stem."""
    store, _, _ = build_fixture_store([
        "regress/bug_d/server.py",
        "regress/bug_d/graph_facts.py",
    ])
    pairs = near_miss_pairs(*orphans(store, flavour="path"))
    hit = [nm for nm in pairs
           if nm.producer.match_text == "graph.ladybug" and nm.consumer.match_text == "graph.kuzu"]
    assert hit, f"BUG D pair missing; pairs were {[(p.producer.match_text, p.consumer.match_text) for p in pairs]}"
    assert hit[0].shared_stem == "graph"
