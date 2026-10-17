"""`cg writes-dead` — two checks over NAME/WRITES_NAME/READS_NAME, reported
separately because their false-positive rates are wildly different.

(a) DANGLING-GLOBAL: an ident declared `global` inside some function of
    module M, where M has no module-level binding (assign/def/import) of
    that ident anywhere. scopes.py already computed this exactly —
    `binding_kind` contains "global-only" precisely when a `global`
    statement named the ident and nothing in the module's own top-level
    body ever bound it — so this check is a lookup, not a heuristic.
    THIS IS BUG C: mcp/graph_tools.py's `global _symbol_index_cache,
    _symbol_index_count` binds a slot graph_tools.py never defines at
    module level; the real slot is module-level in mcp/ladybug_ops.py. The
    diagnosis names that real slot by finding every OTHER module's NAME
    node with the same ident that IS module-level-bound — near-zero false
    positives, because a typo or a stale cross-module assumption is almost
    the only way to end up here.

(b) WRITE-WITH-NO-READ: a NAME slot with >=1 WRITES_NAME edge and 0
    READS_NAME edges anywhere in the corpus. Much weaker: a slot read only
    via `getattr(mod, "x")`, `from mod import *`, or only from test code
    (excluded from the default corpus) reads as a false positive here.
    Reported as a triage list, never a diagnosis — see docs/LIMITS.md for
    the measured false-positive rate on this corpus.

stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..model import Edge, GraphStore, Node


def _ident_of(name_id: str) -> str:
    return name_id.rsplit("::", 1)[-1]


def _in_scope(node: Node, scope_prefixes: Optional[list[str]]) -> bool:
    if not scope_prefixes:
        return True
    return any((node.path or "").startswith(p) for p in scope_prefixes)


@dataclass
class DanglingGlobalFinding:
    slot: Node
    global_writes: list[Edge]              # WRITES_NAME edges with via="global-stmt"
    real_slots: list[Node] = field(default_factory=list)   # module-level-bound NAME nodes,
                                                             # same ident, a DIFFERENT module


@dataclass
class WriteNoReadFinding:
    slot: Node
    writes: list[Edge]


def _module_level_bound_index(store: GraphStore) -> dict[str, list[Node]]:
    """ident -> every NAME node, across the whole corpus, that IS bound at
    module level (assign/def/import) — the candidate pool for "where does
    the real slot live" in a dangling-global diagnosis."""
    idx: dict[str, list[Node]] = {}
    for n in store.nodes_of_kind("NAME"):
        bk = set(n.attrs.get("binding_kind", []))
        if bk & {"assign", "def", "import"}:
            idx.setdefault(_ident_of(n.id), []).append(n)
    return idx


def dangling_globals(store: GraphStore, scope_prefixes: Optional[list[str]] = None) -> list[DanglingGlobalFinding]:
    real_slot_index = _module_level_bound_index(store)
    out: list[DanglingGlobalFinding] = []
    for n in store.nodes_of_kind("NAME"):
        if "global-only" not in set(n.attrs.get("binding_kind", [])):
            continue
        if not _in_scope(n, scope_prefixes):
            continue
        ident = _ident_of(n.id)
        writes = [e for e in store.in_edges(n.id, "WRITES_NAME") if e.attrs.get("via") == "global-stmt"]
        real = sorted(
            (rn for rn in real_slot_index.get(ident, []) if rn.path != n.path),
            key=lambda rn: (rn.path or "", rn.line or 0),
        )
        out.append(DanglingGlobalFinding(slot=n, global_writes=writes, real_slots=real))
    out.sort(key=lambda f: (f.slot.path or "", f.slot.line or 0))
    return out


def write_no_read(store: GraphStore, scope_prefixes: Optional[list[str]] = None) -> list[WriteNoReadFinding]:
    out: list[WriteNoReadFinding] = []
    for n in store.nodes_of_kind("NAME"):
        if not _in_scope(n, scope_prefixes):
            continue
        writes = store.in_edges(n.id, "WRITES_NAME")
        if not writes:
            continue
        if store.in_edges(n.id, "READS_NAME"):
            continue
        out.append(WriteNoReadFinding(slot=n, writes=writes))
    out.sort(key=lambda f: (f.slot.path or "", f.slot.line or 0))
    return out


def read_by_tests(idents: set[str], test_sources) -> dict[str, list[str]]:
    """Cheap `--include-tests` mitigation for the weak check: a whole-word
    textual scan of test file SOURCE (not a full parse/resolve — tests are
    intentionally excluded from the modelled corpus, see config.py) for
    each candidate ident. Returns ident -> sorted list of test rel_paths
    where the token appears at least once. A textual hit is not proof the
    ident is genuinely read there (it could be a coincidental substring
    match on a docstring, or the ident used as an unrelated local), which
    is exactly why this is a `--include-tests` RE-CHECK, not baked into the
    default write-with-no-read result."""
    import re
    hits: dict[str, list[str]] = {ident: [] for ident in idents}
    patterns = {ident: re.compile(r"\b" + re.escape(ident) + r"\b") for ident in idents}
    for sf in test_sources:
        for ident, pat in patterns.items():
            if pat.search(sf.source):
                hits[ident].append(sf.rel_path)
    return {ident: sorted(paths) for ident, paths in hits.items() if paths}
