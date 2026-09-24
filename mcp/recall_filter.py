"""Shared retracted/superseded filter for every recall path.

``memory_retract`` only appends tombstones to ``retractions.jsonl``; the
finding's own Qdrant point and Mnemosyne memory stay in place. Every lane
that hands stored findings back to an agent (investigation_search,
rag_context_search, memory_surface, ground, consolidate, causal edges) must
therefore drop retracted findings itself, by finding id, and say how many it
dropped. This module is that one filter, so the lanes cannot drift apart.

Malformed investigation directories (a name that fails today's id validation,
such as a legacy ``undefined`` dir, or one that cannot be read) are handled
per directory and reported. They never disable filtering for everything else.
Reads are path-level and read-only: nothing here creates, renames or deletes
a directory.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

from inv_store import (
    _RESOLUTION_STATES,
    _fold_retracted_ids,
    _read_jsonl,
    _validated_investigation_id,
)

logger = logging.getLogger("loci-mcp.recall_filter")


def _safe_child(root: Path, name: str) -> Optional[Path]:
    """``root/name`` when ``name`` is a single existing child directory, else None."""
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\x00" in name:
        return None
    path = root / name
    try:
        if not path.is_dir():
            return None
        if path.resolve().parent != root.resolve():
            return None
    except OSError:
        return None
    return path


# Per-investigation cache for _final_resolutions, keyed on the two logs' stat.
# rag_context_search / memory_surface (and so every ground() call) otherwise
# re-parse findings.jsonl on each query just to find superseded ids.
_RESOLUTIONS_CACHE: dict[str, tuple[tuple, dict[str, str]]] = {}
_RESOLUTIONS_CACHE_MAX = 512


def _stat_key(path: Path) -> Optional[tuple]:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def _final_resolutions(inv_path: Path) -> dict[str, str]:
    """``{finding_id: resolution}`` from findings.jsonl then finding_updates.jsonl (last wins).

    Cached per directory until either log's inode, size or mtime changes. The
    returned dict is shared: callers must not mutate it.
    """
    key = (_stat_key(inv_path / "findings.jsonl"), _stat_key(inv_path / "finding_updates.jsonl"))
    cached = _RESOLUTIONS_CACHE.get(str(inv_path))
    if cached is not None and cached[0] == key:
        return cached[1]
    out = _read_final_resolutions(inv_path)
    if len(_RESOLUTIONS_CACHE) >= _RESOLUTIONS_CACHE_MAX:
        _RESOLUTIONS_CACHE.clear()
    _RESOLUTIONS_CACHE[str(inv_path)] = (key, out)
    return out


_TEXTS_CACHE: dict[str, tuple[tuple, set[str]]] = {}


def _retracted_texts(inv_path: Path, rids: set[str]) -> set[str]:
    """Texts of the retracted findings, cached until findings.jsonl or the id set changes."""
    key = (_stat_key(inv_path / "findings.jsonl"), frozenset(rids))
    cached = _TEXTS_CACHE.get(str(inv_path))
    if cached is not None and cached[0] == key:
        return cached[1]
    texts: set[str] = set()
    for f in _read_jsonl(inv_path / "findings.jsonl"):
        if isinstance(f, dict) and str(f.get("id", "")) in rids:
            t = str(f.get("text", "") or "").strip()
            if t:
                texts.add(t)
    if len(_TEXTS_CACHE) >= _RESOLUTIONS_CACHE_MAX:
        _TEXTS_CACHE.clear()
    _TEXTS_CACHE[str(inv_path)] = (key, texts)
    return texts


def _read_final_resolutions(inv_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in _read_jsonl(inv_path / "findings.jsonl"):
        if not isinstance(f, dict) or f.get("record_type") == "access":
            continue
        fid = str(f.get("id") or "")
        if fid and f.get("resolution"):
            out[fid] = str(f.get("resolution")).lower()
    for rec in _read_jsonl(inv_path / "finding_updates.jsonl"):
        if not isinstance(rec, dict) or rec.get("record_type") != "resolution":
            continue
        fid = str(rec.get("finding_id") or "")
        res = str(rec.get("resolution") or "").lower()
        if fid and res in _RESOLUTION_STATES:
            out[fid] = res
    return out


class RecallFilter:
    """Retracted and superseded finding ids for a set of investigations."""

    def __init__(self) -> None:
        self.retracted: dict[str, set[str]] = {}
        self.retracted_texts: dict[str, set[str]] = {}
        self.superseded: dict[str, set[str]] = {}
        self.skipped: list[dict] = []
        self.malformed: list[dict] = []

    @property
    def all_retracted(self) -> set[str]:
        out: set[str] = set()
        for ids in self.retracted.values():
            out |= ids
        return out

    @staticmethod
    def row_finding_id(row: dict) -> str:
        return str(row.get("finding_id") or row.get("id") or "")

    def is_retracted(self, row: dict) -> bool:
        """True when a row names, or repeats the text of, a retracted finding.

        A row whose index payload carries ``retracted: true`` (set on the Qdrant
        point by memory_retract, cleared by memory_restore) is retracted even
        when that investigation's log could not be read or is not in scope.
        """
        if row.get("retracted") is True:
            return True
        if not self.retracted:
            return False
        inv = str(row.get("investigation_id") or "")
        fid = self.row_finding_id(row)
        if inv:
            if fid and fid in self.retracted.get(inv, ()):
                return True
            text = str(row.get("text") or "").strip()
            return bool(text) and text in self.retracted_texts.get(inv, ())
        # No investigation on the row: finding ids are uuids, match across scope.
        return bool(fid) and fid in self.all_retracted

    def is_superseded(self, row: dict) -> bool:
        inv = str(row.get("investigation_id") or "")
        fid = self.row_finding_id(row)
        return bool(inv and fid) and fid in self.superseded.get(inv, ())

    @classmethod
    def finding_key(cls, row: dict) -> str:
        """Stable per-finding key so duplicate rows of one finding count once."""
        fid = cls.row_finding_id(row)
        inv = str(row.get("investigation_id") or "")
        return f"{inv}|{fid}" if fid else f"{inv}|text:{str(row.get('text') or '').strip()[:220]}"

    def split(self, rows: Iterable[dict]) -> tuple[list[dict], set[str]]:
        """Drop retracted rows; return ``(kept, distinct excluded finding keys)``."""
        kept: list[dict] = []
        excluded: set[str] = set()
        for row in rows:
            if isinstance(row, dict) and self.is_retracted(row):
                excluded.add(self.finding_key(row))
                continue
            kept.append(row)
        return kept, excluded

    def status(self) -> dict:
        """Report block for tool responses: ``ok`` unless some directory was not filtered."""
        out: dict = {"status": "degraded" if self.skipped else "ok"}
        if self.skipped:
            out["skipped_investigations"] = list(self.skipped)
        if self.malformed:
            out["malformed_investigations"] = list(self.malformed)
        return out


def build_recall_filter(
    root: Path,
    investigation_ids: Optional[Iterable[str]] = None,
    *,
    with_texts: bool = True,
    with_superseded: bool = False,
) -> RecallFilter:
    """Build a :class:`RecallFilter` over ``investigation_ids`` (all dirs under ``root`` when None).

    Per-directory fail-safe: a directory whose name fails id validation is
    still folded (by path) and listed in ``malformed``; one that cannot be read
    is listed in ``skipped`` and the rest are still filtered.
    """
    rf = RecallFilter()
    root = Path(root)
    if investigation_ids is None:
        try:
            names = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
        except OSError as exc:
            rf.skipped.append({"investigation_id": "*", "reason": f"cannot list memory dir: {exc}"})
            return rf
    else:
        names = sorted({str(n) for n in investigation_ids if n})
    for name in names:
        inv_path = _safe_child(root, name)
        if inv_path is None:
            continue  # no such investigation on disk: nothing retracted there
        try:
            _validated_investigation_id(name)
        except ValueError as exc:
            rf.malformed.append({"investigation_id": name, "reason": str(exc)})
        try:
            rids = _fold_retracted_ids(inv_path / "retractions.jsonl")
            if rids:
                rf.retracted[name] = rids
            if with_texts and rids:
                rf.retracted_texts[name] = _retracted_texts(inv_path, rids)
            if with_superseded:
                sup = {fid for fid, res in _final_resolutions(inv_path).items() if res == "superseded"}
                if sup:
                    rf.superseded[name] = sup
        except Exception as exc:  # per-dir: one unreadable dir never disables the rest
            logger.warning("recall filter: skipping investigation dir %r: %r", name, exc)
            rf.skipped.append({"investigation_id": name, "reason": repr(exc)})
    return rf
