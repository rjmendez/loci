"""On-disk investigation storage, split from server.py.

This module owns per-investigation directory layout, JSONL append/read helpers,
manifest load/save with a write-through cache, and small pure helpers shared by
read paths.

The memory root is injected, not imported: server.py passes a lambda closing
over its own global so tests that rebind that root still steer every write.
This module keeps no default root because one missed reference would silently
hit the operator's real store while fail-open behavior hid the mistake.
Everything goes through ``_root()``.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("loci-mcp")

_get_memory_dir = None  # injected by register(); returns the memory root Path


def _parse_lock_timeout() -> float:
    raw = str(os.environ.get("LOCI_STORE_LOCK_TIMEOUT_S", "1.5") or "").strip()
    try:
        value = float(raw)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    return 1.5


_STORE_LOCK_TIMEOUT_S = _parse_lock_timeout()
_STORE_LOCK_INITIAL_BACKOFF_S = 0.01
_STORE_LOCK_MAX_BACKOFF_S = 0.1


class StoreBusyError(TimeoutError):
    """Bounded advisory-lock acquisition timed out; caller should retry."""

    def __init__(self, path: Path, *, timeout_s: float, operation: str):
        self.path = Path(path)
        self.timeout_s = float(timeout_s)
        self.operation = str(operation)
        super().__init__(
            f"{self.operation} lock busy for {self.path.name}; retry "
            f"(waited {self.timeout_s:.2f}s)"
        )


_MAX_INVESTIGATION_ID_BYTES = 255


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

# The numeric weight each confidence label carries into a derived_from chain product.
# Lives here, not in server.py, because investigation_tools reads the same chain and
# must not import server.
_CONFIDENCE_TO_NUMERIC: dict[str, float] = {"high": 0.9, "medium": 0.6, "low": 0.3}
_NEUTRAL_NUMERIC_CONFIDENCE = 0.6


def _node_numeric_confidence(node: dict) -> float:
    """The confidence a stored finding contributes to a derived_from chain product.

    Absence is not certainty. Missing ``numeric_confidence`` must not become
    ``1.0``, or the record reads as perfect certainty and stops constraining the
    chain product. Fall back to the record's confidence label, then to the same
    neutral ``0.6`` ``_store_numeric_confidence`` uses when no label exists.
    """
    nc = node.get("numeric_confidence")
    if nc is not None:
        try:
            return max(0.0, min(1.0, float(nc)))
        except (TypeError, ValueError):
            pass
    label = str(node.get("confidence") or "").strip().lower()
    return _CONFIDENCE_TO_NUMERIC.get(label, _NEUTRAL_NUMERIC_CONFIDENCE)


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


def _distinctive_entity_set(entities: dict | None) -> set[str]:
    """Flatten the server's typed-entity dict to a set of distinctive entity tokens."""
    out: set[str] = set()
    if not isinstance(entities, dict):
        return out
    for bucket in ("ips", "hashes", "cves", "emails", "hostnames"):
        for v in entities.get(bucket, []) or []:
            s = str(v).strip().lower()
            if s:
                out.add(s)
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_INVALID_INVESTIGATION_ID_SENTINELS = frozenset({"undefined", "null", "none"})

# Most filesystems (ext4, APFS, NTFS) cap a single path component at 255 bytes;
# an id at or beyond that raises an OS-level ENAMETOOLONG from Path.mkdir()
# instead of our own clean ValueError. Reject it up front so every caller sees
# the same validation error regardless of the underlying filesystem.
_MAX_INVESTIGATION_ID_BYTES = 255


def _validated_investigation_id(investigation_id: str) -> str:
    if not isinstance(investigation_id, str):
        raise ValueError(f"Invalid investigation_id: {investigation_id!r}")
    trimmed = investigation_id.strip()
    if not trimmed:
        raise ValueError("Invalid investigation_id: empty string")
    if trimmed.lower() in _INVALID_INVESTIGATION_ID_SENTINELS:
        raise ValueError(
            f"Invalid investigation_id: {investigation_id!r} is a missing-value sentinel"
        )
    if not re.match(r'^[A-Za-z0-9_\-]+$', trimmed):
        raise ValueError(f'Invalid investigation_id: {investigation_id!r}')
    if len(os.fsencode(trimmed)) > _MAX_INVESTIGATION_ID_BYTES:
        raise ValueError(f"Invalid investigation_id: {investigation_id!r} exceeds {_MAX_INVESTIGATION_ID_BYTES} bytes")
    return trimmed


def _inv_dir(investigation_id: str) -> Path:
    investigation_id = _validated_investigation_id(investigation_id)
    root = _root().resolve()
    candidate = (root / investigation_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError('Path escape detected in investigation_id')
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


_manifest_cache: dict[str, str] = {}  # investigation_id → raw JSON string (write-through)
_MANIFEST_CACHE_MAXSIZE = 256


def _load_manifest(investigation_id: str) -> dict | None:
    investigation_id = _validated_investigation_id(investigation_id)
    raw = _manifest_cache.get(investigation_id)
    if raw is None:
        p = _root() / investigation_id / "manifest.json"
        if not p.exists():
            return None
        raw = p.read_text()
        if len(_manifest_cache) >= _MANIFEST_CACHE_MAXSIZE:
            _manifest_cache.pop(next(iter(_manifest_cache)))
        _manifest_cache[investigation_id] = raw
    manifest = json.loads(raw)
    # Backward compat: initialize ACL fields if missing (old investigations)
    if "owner" not in manifest:
        manifest["owner"] = ""
    if "acl" not in manifest:
        manifest["acl"] = []
    return manifest


def _load_manifest_fresh(investigation_id: str) -> dict | None:
    """Load the manifest from disk without consulting the write-through cache."""
    investigation_id = _validated_investigation_id(investigation_id)
    p = _root() / investigation_id / "manifest.json"
    if not p.exists():
        return None
    raw = p.read_text()
    manifest = json.loads(raw)
    if "owner" not in manifest:
        manifest["owner"] = ""
    if "acl" not in manifest:
        manifest["acl"] = []
    return manifest


def _atomic_write_text(path: Path, data: str) -> None:
    """Write ``data`` to ``path`` atomically via a same-directory temp file.

    On failure, remove the temp file and re-raise so no partial or stray file is
    left behind.
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
        except Exception as exc:
            logger.debug("_atomic_write_text: fail-open swallow: %r", exc)
        raise


def _save_manifest(manifest: dict) -> None:
    manifest["updated_at"] = _now()
    p = _inv_dir(manifest["id"]) / "manifest.json"
    data = json.dumps(manifest, indent=2)
    _atomic_write_text(p, data)
    if len(_manifest_cache) >= _MANIFEST_CACHE_MAXSIZE:
        _manifest_cache.pop(next(iter(_manifest_cache)))
    _manifest_cache[manifest["id"]] = data  # keep cache in sync with what we wrote


def _acquire_file_lock(fd: int, path: Path, *, exclusive: bool, timeout_s: float | None = None) -> None:
    """Bounded advisory lock with short exponential backoff.

    Raises ``StoreBusyError`` when contention outlives the timeout so callers
    can return a retryable ``"busy"`` result instead of hanging until the
    transport dies.
    """
    timeout = _STORE_LOCK_TIMEOUT_S if timeout_s is None else float(timeout_s)
    deadline = time.monotonic() + timeout
    wait_s = _STORE_LOCK_INITIAL_BACKOFF_S
    mode = (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB
    while True:
        try:
            fcntl.flock(fd, mode)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise StoreBusyError(
                    path,
                    timeout_s=timeout,
                    operation="exclusive" if exclusive else "shared",
                ) from None
            time.sleep(wait_s)
            wait_s = min(wait_s * 2, _STORE_LOCK_MAX_BACKOFF_S)


@contextmanager
def _locked_file(path: Path, mode: str, *, exclusive: bool = True, timeout_s: float | None = None):
    """Open ``path`` and hold a bounded advisory lock for the caller's critical section."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as fh:
        _acquire_file_lock(fh.fileno(), path, exclusive=exclusive, timeout_s=timeout_s)
        try:
            yield fh
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception as exc:
                logger.debug("_locked_file: fail-open swallow: %r", exc)


def _append_jsonl(path: Path, entry: dict) -> None:
    # Exclusive advisory lock around the append so concurrent writers (e.g. parallel
    # workflow agents recording to the same investigation) can't interleave a >PIPE_BUF
    # line and corrupt the file. Bound the wait so a contended writer returns a
    # retryable busy signal before the MCP transport times out and drops the response.
    line = json.dumps(entry) + "\n"
    with _locked_file(path, "a", exclusive=True) as f:
        f.write(line)
        f.flush()


# rag_context_search access bookkeeping lives in its own per-investigation log.
# Older builds appended it to findings.jsonl under the finding's own id, so any
# reader where the last row wins saw a text-less access row instead of the finding.
FINDINGS_LOG_NAME = "findings.jsonl"
ACCESS_LOG_NAME = "access.jsonl"
_ACCESS_RECORD_TYPES = frozenset({"access"})


def _is_access_row(rec) -> bool:
    """True for an access-bookkeeping row. Such a row is not a finding."""
    return (isinstance(rec, dict)
            and (rec.get("record_type") or rec.get("type") or "") in _ACCESS_RECORD_TYPES)


def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL log. Legacy access rows are dropped from findings.jsonl."""
    if not path.exists():
        return []
    out = []
    bad = 0
    drop_access = path.name == FINDINGS_LOG_NAME
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rec = json.loads(line)
            except Exception:
                bad += 1
                continue
            if drop_access and _is_access_row(rec):
                continue
            out.append(rec)
    if bad:
        logger.debug("_read_jsonl: skipped %d unparseable line(s) in %s", bad, path)
    return out


def _rewrite_jsonl_preserving(path: Path, update) -> int:
    """Atomically rewrite a JSONL log, changing only the rows ``update`` replaces.

    ``update(rec)`` receives each parsed dict row and returns either a
    replacement dict or None to keep the row as it is. All other lines are
    written back byte-for-byte. That includes lines that do not parse and legacy
    findings.jsonl access rows, because a rewrite must never be what deletes a
    record. Returns the number of rows replaced.
    """
    skip_access = path.name == FINDINGS_LOG_NAME
    new_lines = []
    replaced = 0
    for line in path.read_text().splitlines():
        stripped = line.strip()
        rec = None
        if stripped:
            try:
                rec = json.loads(stripped)
            except Exception:
                rec = None
        if isinstance(rec, dict) and not (skip_access and _is_access_row(rec)):
            new = update(rec)
            if new is not None:
                new_lines.append(json.dumps(new))
                replaced += 1
                continue
        new_lines.append(line)
    _atomic_write_text(path, "\n".join(new_lines) + ("\n" if new_lines else ""))
    return replaced


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


# Corroboration evidence carried alongside a dense-similarity score. Every lane
# builds its refs through _make_ref, so the passthrough lives here rather than at
# the one call site that needs it; lexical records simply carry none of these.
_REF_EVIDENCE_KEYS = (
    "lexical_overlap",
    "pool_median",
    "pool_size",
    "margin",
    "evidence_provenance_tier",
    "provenance_defaulted",
)


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
    for key in _REF_EVIDENCE_KEYS:
        if record.get(key) is not None:
            ref[key] = record[key]
    if "margin" not in ref and ref.get("score") is not None and record.get("pool_median") is not None:
        try:
            ref["margin"] = round(ref["score"] - float(record["pool_median"]), 4)
        except (TypeError, ValueError):
            pass
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
