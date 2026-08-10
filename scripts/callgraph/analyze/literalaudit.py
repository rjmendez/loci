"""Producer/consumer tables, orphan detection, and the fenced near-miss
pairing behind `cg literals`.

THIS IS BUG D: `mcp/server.py:261`'s `LadybugStore(str(MEMORY_DIR /
"graph.ladybug"))` PRODUCES a PATHEXPR whose tail literal is
`graph.ladybug`, with zero CONSUMES anywhere in the corpus; a sibling
`scripts/graph_facts.py:33`'s `glob.glob(os.path.expanduser("~/.hermes/**/
graph.kuzu"), recursive=True)` CONSUMES the bare literal
`~/.hermes/*/graph.kuzu` (normalized), with zero PRODUCES. Both are
"orphans" — a literal with occurrences only in one direction. Neither half
proves the other is the bug; the orphan pair is the LEAD, and --near-miss
is only the convenience of printing them next to each other because they
share a stem ("graph") across a module boundary.

Matching is keyed on `tail` text (LITERAL's own tail segment, or a
PATHEXPR's `tail_literal`) rather than node identity, precisely so a
PATHEXPR producer and a bare-literal consumer with the same tail text are
recognized as the same "thing" even though they're different node kinds.

NEAR_MISS is capped at PROBABLE and never influences reachability or
dead-code — see docs/LIMITS.md.

stdlib only.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Optional

from ..model import Confidence, Edge, GraphStore, Node


@dataclass
class LiteralSite:
    node_id: str            # LITERAL or PATHEXPR id
    match_text: str         # the tail text sites are grouped/matched on
    flavour: str
    direction: str          # "produce" | "consume"
    src: str                # FUNCTION|MODULE id
    path: Optional[str]
    line: Optional[int]
    col: Optional[int]
    role: str


def _dest_match_text(node: Node) -> Optional[str]:
    if node.kind == "LITERAL":
        return node.attrs.get("tail") or node.attrs.get("normalized")
    if node.kind == "PATHEXPR":
        return node.attrs.get("tail_literal")
    return None


def _dest_flavour(node: Node) -> str:
    # PATHEXPR is composed from segments (BinOp `/`, os.path.join, f-string,
    # `+` concat) that CAN be used in a key-lookup position too (e.g. an
    # f-string dict key) — its flavour is stamped by extract/literals.py at
    # the call site that first classified it, same field name as LITERAL.
    return node.attrs.get("flavour", "path")


def collect_sites(store: GraphStore, flavour: Optional[str] = None) -> list[LiteralSite]:
    out: list[LiteralSite] = []
    for kind, direction in (("PRODUCES_LITERAL", "produce"), ("CONSUMES_LITERAL", "consume")):
        for e in store.edges_of_kind(kind):
            node = store.get(e.dst)
            if node is None or node.kind not in ("LITERAL", "PATHEXPR"):
                continue
            text = _dest_match_text(node)
            if text is None:
                continue
            node_flavour = _dest_flavour(node)
            if flavour and node_flavour != flavour:
                continue
            src_node = store.get(e.src)
            out.append(LiteralSite(
                node_id=node.id, match_text=text, flavour=node_flavour, direction=direction,
                src=e.src, path=(src_node.path if src_node is not None else None),
                line=e.attrs.get("line"), col=e.attrs.get("col"), role=e.attrs.get("role", "?"),
            ))
    return out


@dataclass
class LiteralGroup:
    match_text: str
    flavour: str
    produce_sites: list[LiteralSite] = field(default_factory=list)
    consume_sites: list[LiteralSite] = field(default_factory=list)

    @property
    def sites(self) -> list[LiteralSite]:
        return self.produce_sites + self.consume_sites


def group_sites(sites: list[LiteralSite]) -> dict[tuple[str, str], LiteralGroup]:
    groups: dict[tuple[str, str], LiteralGroup] = {}
    for s in sites:
        key = (s.match_text, s.flavour)
        g = groups.setdefault(key, LiteralGroup(match_text=s.match_text, flavour=s.flavour))
        (g.produce_sites if s.direction == "produce" else g.consume_sites).append(s)
    return groups


def literal_table(store: GraphStore, flavour: Optional[str] = None) -> list[LiteralGroup]:
    groups = group_sites(collect_sites(store, flavour))
    return sorted(groups.values(), key=lambda g: g.match_text)


def orphans(store: GraphStore, flavour: Optional[str] = None) -> tuple[list[LiteralGroup], list[LiteralGroup]]:
    """Returns (producer_only, consumer_only) — path-like literals (by
    default: pass flavour="path"/"key" to narrow) with occurrences in
    exactly one direction."""
    groups = group_sites(collect_sites(store, flavour))
    producer_only = sorted(
        (g for g in groups.values() if g.produce_sites and not g.consume_sites),
        key=lambda g: g.match_text,
    )
    consumer_only = sorted(
        (g for g in groups.values() if g.consume_sites and not g.produce_sites),
        key=lambda g: g.match_text,
    )
    return producer_only, consumer_only


def _stem(tail: str) -> str:
    base = tail.rstrip("/").rsplit("/", 1)[-1]
    if "." in base and not base.startswith("."):
        return base.rsplit(".", 1)[0].lower()
    return base.lower()


@dataclass
class NearMiss:
    producer: LiteralGroup
    consumer: LiteralGroup
    shared_stem: str
    distance: float


def near_miss_pairs(producer_only: list[LiteralGroup], consumer_only: list[LiteralGroup]) -> list[NearMiss]:
    """Fenced pairing: same stem (tail text minus extension, case-folded),
    cross-module, capped at PROBABLE, and — per the design's own rule —
    never wired into reachability or dead-code. `distance` is a plain
    difflib ratio, reported as a lead's supporting detail, not a threshold
    this function filters on beyond the stem match."""
    out: list[NearMiss] = []
    for p in producer_only:
        p_stem = _stem(p.match_text)
        p_paths = {s.path for s in p.produce_sites}
        for c in consumer_only:
            if _stem(c.match_text) != p_stem:
                continue
            if p.match_text == c.match_text:
                continue  # exact match is not a "near" miss — it wouldn't be an orphan pair anyway
            c_paths = {s.path for s in c.consume_sites}
            if p_paths <= c_paths and c_paths <= p_paths:
                continue  # same module(s) only — design requires a module boundary
            ratio = difflib.SequenceMatcher(None, p.match_text, c.match_text).ratio()
            out.append(NearMiss(producer=p, consumer=c, shared_stem=p_stem, distance=round(ratio, 3)))
    out.sort(key=lambda nm: (-nm.distance, nm.shared_stem))
    return out


def materialize_near_miss_edges(store: GraphStore, pairs: list[NearMiss]) -> list[Edge]:
    """Adds the derived NEAR_MISS LITERAL->LITERAL/PATHEXPR edges to the
    live store so `cg explain` can show them too. Idempotent-ish: callers
    (the CLI) call this once per invocation on a freshly built store, so
    duplicate edges across repeated calls within one process are the only
    risk, which this function does not need to guard against here."""
    edges: list[Edge] = []
    for nm in pairs:
        for p_site in nm.producer.produce_sites:
            for c_site in nm.consumer.consume_sites:
                e = Edge(src=p_site.node_id, dst=c_site.node_id, kind="NEAR_MISS", confidence=Confidence.PROBABLE,
                         attrs={"distance": nm.distance, "shared_stem": nm.shared_stem,
                                "producer_site": f"{p_site.path}:{p_site.line}",
                                "consumer_site": f"{c_site.path}:{c_site.line}"})
                store.add_edge(e)
                edges.append(e)
    return edges
