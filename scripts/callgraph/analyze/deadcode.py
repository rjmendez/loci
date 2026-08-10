"""`cg dead`'s verdicts. Deliberately conservative: a function is reported
only if it is unreachable under the MOST generous closure this tool can
build (PROBABLE floor — proven + probable). See analyze/reach.py's own
docstring for why every MODULE is treated as a trivial root.

Only two verdicts are implemented this slice — UNREACHABLE and (a stub for)
TEST-ONLY/REACHABLE-ONLY-VIA-UNPROVEN are left for a later slice once test
roots and the probable-tier CALLS rungs (4-6) exist; every row here is
already reachable-under-PROBABLE-or-better, so nothing is misreported as
dead, it is simply less finely triaged than the full design calls for.

Per the design's own rule: never wire this to a non-zero exit code, and
never delete on the strength of this alone — the footer says so on every
run.
"""
from __future__ import annotations

from typing import Optional

from ..model import Confidence, GraphStore, Node
from .reach import reachable_functions


def _is_protocol_dunder(node: Node) -> bool:
    """`__enter__`, `__exit__`, `__init__`, `__iter__`, `__len__`, ... are
    invoked by LANGUAGE MACHINERY (`with`, `for`, `len()`, an operator, a
    constructor call), not by name, so no call graph will ever find a
    reference to them and every one of them is noise on a dead-code list.

    Validated at HEAD: 5 of the 71 rows were protocol dunders, including
    `scripts/glymphatic_sweep.py`'s `_Mutex.__enter__`/`__exit__`, which are
    invoked one screen away by `with _Mutex(MUTEX_FLAG):`.

    A never-instantiated class is a real thing to want to know, but the right
    unit for that is the CLASS, not its `__init__` — reporting the method
    tells the reader something misleading about the method.
    """
    name = node.attrs.get("qualname", "").rsplit(".", 1)[-1]
    return name.startswith("__") and name.endswith("__") and len(name) > 4


def dead_functions(store: GraphStore, scope_prefixes: Optional[list[str]] = None) -> list[Node]:
    reachable = reachable_functions(store, conf_floor=Confidence.PROBABLE)
    out: list[Node] = []
    for n in store.nodes_of_kind("FUNCTION"):
        if n.attrs.get("is_lambda"):
            continue  # a lambda's "identity" is its call site; not a useful dead-code unit on its own
        if _is_protocol_dunder(n):
            continue
        if n.id in reachable:
            continue
        if scope_prefixes and not any((n.path or "").startswith(p) for p in scope_prefixes):
            continue
        out.append(n)
    return out


def registered_but_dead(store: GraphStore, scope_prefixes: Optional[list[str]] = None) -> list[Node]:
    """The hard-gate check: every function some REGISTERS edge names as
    externally callable, that this build's `dead_functions` would still
    report dead. Must be empty on the real corpus — see docs/LIMITS.md and
    build_steps' step 5 gate."""
    dead_ids = {n.id for n in dead_functions(store, scope_prefixes)}
    registered_ids = {e.dst for e in store.edges_of_kind("REGISTERS")}
    return [store.get(nid) for nid in sorted(registered_ids & dead_ids) if store.get(nid) is not None]
