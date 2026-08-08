"""Investigation storage layer — split out of server.py (P2b of the Loci self-review).

Everything here is the on-disk investigation store: the per-investigation directory
layout under the memory root, the JSONL append/read primitives, manifest load/save
(with its write-through cache), and the small pure helpers the read paths share.

The memory root is INJECTED, not imported: server.py passes a lambda closing over
its own memory-root global to ``register()``, so tests that rebind that global to a
tmpdir keep steering every write here. This module deliberately holds no copy of
that root and no module-level default for it — a default would silently resolve to
the operator's real ~/.hermes/memory-sessions the moment one reference was missed,
while fail-open behaviour kept the tests green. Everything goes through _root().
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("loci-mcp")

_get_memory_dir = None  # injected by register(); returns the memory root Path


def _root() -> Path:
    """The investigation memory root, resolved through the injected accessor."""
    return _get_memory_dir()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# Confidence tier ranking — used by investigation_search and _qdrant_similarity_search.
# Defined once here to avoid the same dict appearing inline in multiple functions.
_CONFIDENCE_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


# Finding lifecycle/resolution states. "open" is the default and the implied value for
# any finding record stored before this field existed (absent -> "open"). The three
# resolved states are what exclusion-aware grounding treats as "handled — do not re-report".
_RESOLUTION_STATES: frozenset = frozenset({"open", "fixed", "intentional", "wontfix", "superseded"})


def _summarise_finding(f: dict, *, include_tags: bool = True) -> dict:
    """Compact finding summary for entity-lookup results."""
    out = {
        "finding_id": f.get("id"),
        "investigation_id": f.get("investigation_id"),
        "ts": f.get("ts"),
        "record_type": f.get("record_type") or f.get("type"),
        "confidence": f.get("confidence"),
        "source": f.get("source"),
        "text": str(f.get("text", ""))[:300],
    }
    if include_tags:
        out["tags"] = f.get("tags", [])
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inv_dir(investigation_id: str) -> Path:
    d = _root() / investigation_id
    d.mkdir(parents=True, exist_ok=True)
    return d


_manifest_cache: dict[str, str] = {}  # investigation_id → raw JSON string (write-through)


def _load_manifest(investigation_id: str) -> dict | None:
    raw = _manifest_cache.get(investigation_id)
    if raw is None:
        p = _root() / investigation_id / "manifest.json"
        if not p.exists():
            return None
        raw = p.read_text()
        _manifest_cache[investigation_id] = raw
    manifest = json.loads(raw)
    # Backward compat: initialize ACL fields if missing (old investigations)
    if "owner" not in manifest:
        manifest["owner"] = ""
    if "acl" not in manifest:
        manifest["acl"] = []
    return manifest


def _atomic_write_text(path: Path, data: str) -> None:
    """Write ``data`` to ``path`` atomically via a same-directory temp file.

    The temp file is removed and the error re-raised if anything fails, so a
    failed write never leaves a partial file or stray temp behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(data)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


def _save_manifest(manifest: dict) -> None:
    manifest["updated_at"] = _now()
    p = _inv_dir(manifest["id"]) / "manifest.json"
    data = json.dumps(manifest, indent=2)
    _atomic_write_text(p, data)
    _manifest_cache[manifest["id"]] = data  # keep cache in sync with what we wrote


def _append_jsonl(path: Path, entry: dict) -> None:
    # Exclusive advisory lock around the append so concurrent writers (e.g. parallel
    # workflow agents recording to the same investigation) can't interleave a >PIPE_BUF
    # line and corrupt the file. flock is POSIX-only; degrade to a bare append elsewhere.
    line = json.dumps(entry) + "\n"
    with open(path, "a") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(line)
            f.flush()
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    bad = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                bad += 1
    if bad:
        logger.debug("_read_jsonl: skipped %d unparseable line(s) in %s", bad, path)
    return out


def _finding_updates_path(investigation_id: str) -> Path:
    """Resolution overrides log — scanned by every read path (load/search), so it
    stays SMALL: only finding_resolve appends here (last-write-wins resolutions)."""
    return _inv_dir(investigation_id) / "finding_updates.jsonl"


def _load_resolution_overrides(investigation_id: str) -> dict[str, str]:
    """Return {finding_id: resolution} from finding_updates.jsonl, last-write-wins.

    Only 'resolution' update records with a valid state participate; any other
    record type is skipped cheaply. Verification verdicts live in a separate log
    (finding_verifications.jsonl) so they never bloat this scan. Fail-open: any
    read/parse error yields {} so the read path falls back to stored/default values.
    """
    overrides: dict[str, str] = {}
    try:
        for rec in _read_jsonl(_finding_updates_path(investigation_id)):
            if not isinstance(rec, dict) or rec.get("record_type") != "resolution":
                continue
            fid = str(rec.get("finding_id") or "")
            res = str(rec.get("resolution") or "").lower()
            if fid and res in _RESOLUTION_STATES:
                overrides[fid] = res  # later line wins
    except Exception as exc:  # noqa: BLE001 — never block a read on the overrides log
        logger.debug("resolution overrides load failed (fail-open): %r", exc)
    return overrides


def _load_retracted_ids(investigation_id: str) -> set[str]:
    """Fold ``retractions.jsonl`` into the set of currently-retracted finding ids.

    A finding is retracted iff its id has an ``active:true`` retraction with no
    later ``active:false`` (restore) entry. The log is append-only, so we replay
    it in order and the last entry per finding id wins. Fail-safe: a missing or
    malformed log yields an empty set, never raises.
    """
    path = _inv_dir(investigation_id) / "retractions.jsonl"
    state: dict[str, bool] = {}
    for entry in _read_jsonl(path):
        if not isinstance(entry, dict):
            continue
        fid = entry.get("finding_id")
        if not fid:
            continue
        state[str(fid)] = bool(entry.get("active", True))
    return {fid for fid, active in state.items() if active}


def _make_ref(record: dict, match_type: str, score: float | None = None) -> dict:
    ref = {
        "evidence_id": record.get("evidence_id"),
        "record_type": record.get("record_type"),
        "source": record.get("source"),
        "ts": record.get("ts"),
        "origin": record.get("origin"),
        "snippet": record.get("snippet", ""),
        "match_type": match_type,
    }
    if score is not None:
        ref["score"] = round(score, 4)
    return ref


def _tag_finding_ids(findings: list[dict], investigation_id: str) -> list[dict]:
    """Return shallow copies of findings each carrying a stable ``id``.

    Reuses an existing id field when present, else derives
    ``f"{investigation_id}:{index}"``. Findings on disk are never mutated — this
    only annotates the in-memory copies the checks operate on.
    """
    tagged: list[dict] = []
    for index, f in enumerate(findings or []):
        if not isinstance(f, dict):
            continue
        fid = f.get("id") or f.get("finding_id") or f"{investigation_id}:{index}"
        tagged.append({**f, "id": str(fid)})
    return tagged


def register(get_memory_dir):
    """Inject the memory-root accessor. Must be called before any store call."""
    global _get_memory_dir
    _get_memory_dir = get_memory_dir
