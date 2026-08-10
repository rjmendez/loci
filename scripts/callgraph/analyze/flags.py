"""The partial-assignment ranker behind `cg flags`.

A RANKER, not a detector: every conditional flag in Python matches the
shape "initialized to a constant, reassigned on some paths, escapes" — see
docs/LIMITS.md for the measured rows-per-function rate. Scoped to one
function or file by design; never wire this to a non-zero exit code or a
CI gate (the design's own rule, restated here because it's the one rule
this module could tempt someone into breaking).

THIS IS BUG B: `mcp/grounding.py:ground`'s `degraded` LOCALBINDING has a
constant `False` init, two-to-three guarded reassignments to `True`
(if/except branches), and several `if not S: break`-shaped guard_exits
that skip it entirely — a high ratio of "guards that skip" to "guards that
assign" is exactly the ranking signal.

stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..model import GraphStore, Node


@dataclass
class FlagCandidate:
    binding: Node
    fn_id: str
    escape_form: str
    score: float           # guard_exits skipping / max(1, constant rebinds made)
    pattern: str = "flag"  # "flag" | "accumulator"

    @property
    def ident(self) -> str:
        return self.binding.attrs.get("ident", self.binding.id.rsplit("::", 1)[-1])


def _in_scope(node: Node, scope_prefixes: Optional[list[str]]) -> bool:
    if not scope_prefixes:
        return True
    return any((node.path or "").startswith(p) for p in scope_prefixes)


def classify(binding: Node) -> str:
    """"accumulator" when the local IS reassigned but never by a plain
    rebind-to-a-constant — i.e. every update is `+=`/a for-target/a rebind to
    a computed expression. Counters and running totals live here, and scoring
    them by guard-exit ratio is a false positive by construction: a counter is
    SUPPOSED to be updated on only some paths.

    Measured on this corpus at rev d359e9a (the commit where BUG B was live),
    this is what separated `mcp/grounding.py::ground`'s real `degraded` flag
    from `link_findings`'s `created` and `measure_drift`'s `n_drifted`, which
    both outranked it while being entirely correct code.

    A local with ZERO reassignments is NOT an accumulator — it is a flag that
    is simply never flipped (score 0), which is a different and much weaker
    signal but still a flag by shape.
    """
    contexts = binding.attrs.get("assign_contexts", [])
    if not contexts:
        return "flag"
    forms = set(binding.attrs.get("assign_forms", []))
    if "augmented" in forms:
        # HYBRID counter: `n = 0` ... `n = 1` ... `n += 1`. The constant
        # rebind is a RESET, not a flag flip, so the presence of any `+=`
        # decides. Found by validation: mcp/server.py::_process_reflection_item's
        # `lines_scanned` outranked BUG B itself on the constant-rebind
        # denominator alone (9 guard exits / 1 reset = 9.00).
        return "accumulator"
    if binding.attrs.get("constant_rebind_lines"):
        return "flag"
    return "accumulator"


def rank_flags(store: GraphStore, scope_prefixes: Optional[list[str]] = None,
                fn_filter: Optional[str] = None,
                include_accumulators: bool = False) -> list[FlagCandidate]:
    out: list[FlagCandidate] = []
    for n in store.nodes_of_kind("LOCALBINDING"):
        if not n.attrs.get("init_is_constant"):
            continue
        if not _in_scope(n, scope_prefixes):
            continue
        # local:<fn-id>::<ident> -> fn-id is everything between "local:" and the LAST "::"
        fn_id = n.id[len("local:"):].rsplit("::", 1)[0]
        if fn_filter and fn_id != fn_filter:
            continue
        pattern = classify(n)
        if pattern == "accumulator" and not include_accumulators:
            continue
        guard_exits = n.attrs.get("guard_exits", [])
        # Denominator counts only the rebinds that could actually flip the
        # flag; falling back to assign_lines would re-admit the counter noise
        # this classification exists to remove.
        rebinds = n.attrs.get("constant_rebind_lines", n.attrs.get("assign_lines", []))
        score = len(guard_exits) / max(1, len(rebinds))
        escapes_edges = store.out_edges(fn_id, "ESCAPES")
        escape_form = "?"
        for e in escapes_edges:
            if e.dst == n.id:
                escape_form = e.attrs.get("escape_form", "?")
                break
        out.append(FlagCandidate(binding=n, fn_id=fn_id, escape_form=escape_form,
                                  score=score, pattern=pattern))
    out.sort(key=lambda c: (-c.score, c.binding.path or "", c.binding.line or 0))
    return out
