"""Node/Edge dataclasses, the Confidence enum, and GraphStore.

GraphStore keeps id->Node plus forward/reverse adjacency indexed by edge
kind, so a tiered closure (proven / probable / unproven) is a handful of
cheap dict lookups rather than a full scan-and-filter over every edge. This
is the shared representation every extract/* and analyze/* module reads and
writes; nothing downstream should invent its own dict-of-dicts.

stdlib only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable, Iterator, Optional


class Confidence(IntEnum):
    """Ordered UNPROVEN < PROBABLE < PROVEN so `Confidence.combine(...)` over
    a chain's hops yields the weakest (most honest) link, never the average."""
    UNPROVEN = 0
    PROBABLE = 1
    PROVEN = 2

    @classmethod
    def combine(cls, confidences: Iterable["Confidence"]) -> "Confidence":
        values = list(confidences)
        if not values:
            return cls.PROVEN
        return min(values)

    @classmethod
    def parse(cls, text: str) -> "Confidence":
        return cls[text.strip().upper()]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name.lower()


@dataclass
class Node:
    id: str
    kind: str
    attrs: dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None
    line: Optional[int] = None
    col: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "attrs": self.attrs,
            "path": self.path,
            "line": self.line,
            "col": self.col,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Node":
        return cls(
            id=d["id"], kind=d["kind"], attrs=dict(d.get("attrs") or {}),
            path=d.get("path"), line=d.get("line"), col=d.get("col"),
        )


@dataclass
class Edge:
    src: str
    dst: str
    kind: str
    confidence: Confidence = Confidence.PROVEN
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src, "dst": self.dst, "kind": self.kind,
            "confidence": self.confidence.name, "attrs": self.attrs,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Edge":
        return cls(
            src=d["src"], dst=d["dst"], kind=d["kind"],
            confidence=Confidence.parse(d.get("confidence", "PROVEN")),
            attrs=dict(d.get("attrs") or {}),
        )


class GraphStore:
    """id -> Node, plus forward/reverse adjacency indexed by edge kind."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._out: dict[str, dict[str, list[Edge]]] = {}
        self._in: dict[str, dict[str, list[Edge]]] = {}

    # -- mutation ---------------------------------------------------------

    def add_node(self, node: Node) -> Node:
        """Idempotent: re-adding the same id merges attrs into the existing
        node rather than creating a duplicate (multiple extract passes may
        legitimately touch the same MODULE/NAME node)."""
        existing = self.nodes.get(node.id)
        if existing is not None:
            existing.attrs.update(node.attrs)
            if existing.path is None:
                existing.path = node.path
            if existing.line is None:
                existing.line = node.line
            if existing.col is None:
                existing.col = node.col
            return existing
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: Edge) -> Edge:
        self.edges.append(edge)
        self._out.setdefault(edge.src, {}).setdefault(edge.kind, []).append(edge)
        self._in.setdefault(edge.dst, {}).setdefault(edge.kind, []).append(edge)
        return edge

    def retarget_edge(self, edge: Edge, new_dst: str) -> None:
        """Repoint an already-added edge's dst in place, fixing up the
        reverse index too. Used by extract/registry.py to rewire a
        DECORATED_BY edge from the placeholder EXTERNAL sink defs.py created
        (before a REGISTRY node for that decorator site existed) onto the
        real REGISTRY node, without creating a second, double-counted edge."""
        old_bucket = self._in.get(edge.dst, {}).get(edge.kind)
        if old_bucket is not None:
            old_bucket[:] = [e for e in old_bucket if e is not edge]
        edge.dst = new_dst
        self._in.setdefault(new_dst, {}).setdefault(edge.kind, []).append(edge)

    # -- reads --------------------------------------------------------------

    def get(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.nodes

    def out_edges(self, node_id: str, kind: Optional[str] = None) -> list[Edge]:
        by_kind = self._out.get(node_id, {})
        if kind is None:
            return [e for edges in by_kind.values() for e in edges]
        return list(by_kind.get(kind, []))

    def in_edges(self, node_id: str, kind: Optional[str] = None) -> list[Edge]:
        by_kind = self._in.get(node_id, {})
        if kind is None:
            return [e for edges in by_kind.values() for e in edges]
        return list(by_kind.get(kind, []))

    def nodes_of_kind(self, kind: str) -> Iterator[Node]:
        return (n for n in self.nodes.values() if n.kind == kind)

    def edges_of_kind(self, kind: str) -> Iterator[Edge]:
        return (e for e in self.edges if e.kind == kind)

    def stats(self) -> dict[str, int]:
        node_counts: dict[str, int] = {}
        for n in self.nodes.values():
            node_counts[n.kind] = node_counts.get(n.kind, 0) + 1
        edge_counts: dict[str, int] = {}
        for e in self.edges:
            edge_counts[e.kind] = edge_counts.get(e.kind, 0) + 1
        return {"nodes": len(self.nodes), "edges": len(self.edges), **{
            f"node:{k}": v for k, v in node_counts.items()
        }, **{f"edge:{k}": v for k, v in edge_counts.items()}}

    # -- (de)serialization --------------------------------------------------

    def to_json(self) -> str:
        return json.dumps({
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "GraphStore":
        data = json.loads(text)
        store = cls()
        for nd in data["nodes"]:
            store.add_node(Node.from_dict(nd))
        for ed in data["edges"]:
            store.add_edge(Edge.from_dict(ed))
        return store

    _DOT_SHAPE_BY_KIND = {
        "FUNCTION": "box", "MODULE": "folder", "CLASS": "component",
        "REGISTRY": "hexagon", "ENTRYPOINT": "doublecircle", "NAME": "ellipse",
        "LITERAL": "note", "PATHEXPR": "note",
    }
    _DOT_STYLE_BY_CONFIDENCE = {
        Confidence.PROVEN: "solid", Confidence.PROBABLE: "dashed", Confidence.UNPROVEN: "dotted",
    }

    def to_dot(self, node_ids: Optional[Iterable[str]] = None,
               edges: Optional[Iterable[Edge]] = None) -> str:
        """Graphviz DOT export. With no filter this emits the WHOLE graph
        (every node/edge this build produced — large: tens of thousands of
        nodes for the real corpus, fine for `dot -Tsvg` but not for
        eyeballing directly). Pass `node_ids`/`edges` — e.g. the node ids
        and Edge objects a `cg reach`/`cg callers`/`cg paths` traversal
        already collected — to emit just that subgraph instead, which is
        the normal way to actually look at one of these. Edge style encodes
        confidence (solid=proven, dashed=probable, dotted=unproven) so the
        honesty tiering survives the trip through Graphviz."""
        if node_ids is None:
            sel_nodes = list(self.nodes.values())
        else:
            ids = set(node_ids)
            sel_nodes = [self.nodes[i] for i in ids if i in self.nodes]
        sel_edges = list(self.edges) if edges is None else list(edges)

        def esc(s: object) -> str:
            return str(s).replace("\\", "\\\\").replace('"', '\\"')

        lines = ["digraph callgraph {", "  rankdir=LR;", "  node [shape=box, fontsize=10];"]
        for n in sel_nodes:
            label = esc(n.attrs.get("qualname") or n.attrs.get("match_text") or n.id)
            shape = self._DOT_SHAPE_BY_KIND.get(n.kind, "box")
            lines.append(f'  "{esc(n.id)}" [label="{label}\\n[{n.kind}]", shape={shape}];')
        if "?" not in {n.id for n in sel_nodes} and any(e.dst == "?" or e.src == "?" for e in sel_edges):
            lines.append('  "?" [label="? UNRESOLVED", shape=doublecircle, style=filled, fillcolor=lightgrey];')
        for e in sel_edges:
            style = self._DOT_STYLE_BY_CONFIDENCE.get(e.confidence, "solid")
            lines.append(f'  "{esc(e.src)}" -> "{esc(e.dst)}" [label="{esc(e.kind)}", style={style}];')
        lines.append("}")
        return "\n".join(lines)
