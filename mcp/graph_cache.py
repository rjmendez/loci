"""
Deterministic graph tool result cache.

Unlike the Qdrant-backed memory cache (cache.py), this cache stores
exact deterministic outputs from graph/analytics tools.

Cache key: stable hash of (tool_name, normalized_params, graph_revision).
Storage: JSONL files per tool in LOCI_GRAPH_CACHE_DIR.
Invalidation: on code_graph_ingest, code_memory_relink, or TTL expiry.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

GRAPH_CACHE_DIR = Path(os.getenv("LOCI_GRAPH_CACHE_DIR", "~/.loci/graph_cache")).expanduser()
GRAPH_CACHE_TTL_MIN = float(os.getenv("LOCI_GRAPH_CACHE_TTL_MIN", "30"))


@dataclass(slots=True)
class GraphCacheEntry:
    key: str
    tool_name: str
    params_hash: str
    result: Any
    cached_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    graph_revision: str = ""
    hits: int = 0

    def age_minutes(self) -> float:
        ts = datetime.fromisoformat(self.cached_at)
        return (datetime.now(timezone.utc) - ts).total_seconds() / 60

    def is_valid(self, ttl_minutes: float = GRAPH_CACHE_TTL_MIN) -> bool:
        return self.age_minutes() <= ttl_minutes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "tool_name": self.tool_name,
            "params_hash": self.params_hash,
            "result": self.result,
            "cached_at": self.cached_at,
            "graph_revision": self.graph_revision,
            "hits": self.hits,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphCacheEntry":
        return cls(
            key=d["key"],
            tool_name=d["tool_name"],
            params_hash=d["params_hash"],
            result=d["result"],
            cached_at=d.get("cached_at", ""),
            graph_revision=d.get("graph_revision", ""),
            hits=d.get("hits", 0),
        )


def make_cache_key(tool_name: str, params: Dict[str, Any], graph_revision: str = "") -> str:
    """Stable cache key from tool name + normalized params + graph revision."""
    normalized = json.dumps(params, sort_keys=True, ensure_ascii=True)
    raw = f"{tool_name}|{normalized}|{graph_revision}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class GraphCache:
    def __init__(self, cache_dir: Optional[Path] = None, ttl_minutes: Optional[float] = None):
        self.cache_dir = Path(cache_dir or GRAPH_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_minutes = ttl_minutes if ttl_minutes is not None else GRAPH_CACHE_TTL_MIN

    def _entry_path(self, tool_name: str) -> Path:
        safe = tool_name.replace("/", "_").replace(".", "_")
        return self.cache_dir / f"{safe}.jsonl"

    def get(self, tool_name: str, params: Dict[str, Any], graph_revision: str = "") -> Optional[GraphCacheEntry]:
        """Look up a cached result. Returns None on miss or TTL expiry."""
        key = make_cache_key(tool_name, params, graph_revision)
        path = self._entry_path(tool_name)
        if not path.exists():
            return None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = GraphCacheEntry.from_dict(json.loads(line))
                    if entry.key == key and entry.is_valid(self.ttl_minutes):
                        return entry
                except (json.JSONDecodeError, KeyError):
                    continue
        return None

    def put(self, tool_name: str, params: Dict[str, Any], result: Any, graph_revision: str = "") -> GraphCacheEntry:
        """Store a result. Appends to JSONL; compaction happens on invalidation."""
        key = make_cache_key(tool_name, params, graph_revision)
        entry = GraphCacheEntry(
            key=key,
            tool_name=tool_name,
            params_hash=key,
            result=result,
            graph_revision=graph_revision,
        )
        path = self._entry_path(tool_name)
        with open(path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
        return entry

    def invalidate(self, tool_name: Optional[str] = None) -> int:
        """
        Invalidate cache entries. If tool_name given, only that tool.
        If None, invalidates all (called on code_graph_ingest / code_memory_relink).
        Returns count of files cleared.
        """
        count = 0
        if tool_name:
            path = self._entry_path(tool_name)
            if path.exists():
                path.unlink()
                count = 1
        else:
            for p in self.cache_dir.glob("*.jsonl"):
                p.unlink()
                count += 1
        return count

    def stats(self) -> Dict[str, Any]:
        """Return cache stats: entry counts and file sizes per tool."""
        stats = {}
        for p in self.cache_dir.glob("*.jsonl"):
            tool = p.stem
            with open(p) as f:
                lines = [l for l in f if l.strip()]
            stats[tool] = {"entries": len(lines), "bytes": p.stat().st_size}
        return stats
