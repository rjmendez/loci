"""Write-once sha256 verification stamps for large pinned snapshot files.

Hashing every product of a snapshot on every open is too slow once a snapshot
holds tens of GB (BANC banc_888 is about 29 GB). Adapters use this module so
that file contents are hashed only:

* on an explicit verify (``force=True``), or
* when no valid stamp exists for the file's current ``(size, mtime_ns)``.

A stamp is a small JSON file under ``<snapshot>/manifest/hash-stamps/``. Its
name is derived from ``(relative_path, expected sha256, size_bytes, mtime_ns)``,
so any change to the file's size or mtime, or to the manifest's expected hash,
looks up a different stamp and forces a rehash. A valid stamp is never
rewritten (write-once). Stamps are written only after the recomputed sha256
matched the manifest and the file's stat did not change while it was hashed.

Trust model: a stamp is as trustworthy as the snapshot directory itself. It is
a cache of a previous verification, not a signature. A byte change that keeps
both the size and the nanosecond mtime is not detected by a stamped open; run
an explicit verify for that (``verify_hashes=True`` on the adapters).

No network, no global state; safe to reuse from any adapter.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

HASH_STAMP_SCHEMA_VERSION = "fbh-hash-stamp/v1"
HASH_STAMP_RELATIVE_DIR = "manifest/hash-stamps"

METHOD_HASHED = "hashed"
METHOD_STAMP = "stamp"
METHOD_SIZE_ONLY = "size_only"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class FileStat:
    size_bytes: int
    mtime_ns: int

    @classmethod
    def of(cls, path: Path) -> "FileStat":
        st = os.stat(path)
        return cls(size_bytes=int(st.st_size), mtime_ns=int(st.st_mtime_ns))


@dataclass(frozen=True)
class HashCheck:
    """Result of one file check. ``matched`` is False only on a real mismatch."""

    relative_path: str
    expected_sha256: str
    actual_sha256: str | None
    method: str
    matched: bool
    stamp_written: bool = False
    warning: str | None = None


class HashStampCache:
    """Stamp store rooted at one directory (usually ``<snapshot>/manifest/hash-stamps``)."""

    def __init__(self, stamp_dir: str | Path):
        self.stamp_dir = Path(stamp_dir)

    @classmethod
    def for_snapshot(cls, snapshot_root: str | Path) -> "HashStampCache":
        return cls(Path(snapshot_root) / HASH_STAMP_RELATIVE_DIR)

    @staticmethod
    def stamp_key(relative_path: str, expected_sha256: str, stat: FileStat) -> str:
        material = "\x00".join(
            [
                HASH_STAMP_SCHEMA_VERSION,
                str(relative_path).replace("\\", "/"),
                str(expected_sha256).lower(),
                str(int(stat.size_bytes)),
                str(int(stat.mtime_ns)),
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def stamp_path(self, relative_path: str, expected_sha256: str, stat: FileStat) -> Path:
        return self.stamp_dir / f"{self.stamp_key(relative_path, expected_sha256, stat)}.json"

    def _payload(self, relative_path: str, sha256: str, stat: FileStat) -> dict[str, object]:
        return {
            "schema_version": HASH_STAMP_SCHEMA_VERSION,
            "relative_path": str(relative_path).replace("\\", "/"),
            "sha256": str(sha256).lower(),
            "size_bytes": int(stat.size_bytes),
            "mtime_ns": int(stat.mtime_ns),
        }

    def lookup(self, relative_path: str, expected_sha256: str, stat: FileStat) -> bool:
        """True when a valid stamp exists for exactly this (path, sha, size, mtime_ns)."""
        path = self.stamp_path(relative_path, expected_sha256, stat)
        try:
            if path.is_symlink() or not path.is_file():
                return False
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(data, dict):
            return False
        expected = self._payload(relative_path, expected_sha256, stat)
        return all(data.get(key) == value for key, value in expected.items())

    def record(self, relative_path: str, sha256: str, stat: FileStat) -> bool:
        """Write the stamp once. Returns True when a new stamp was written."""
        if self.lookup(relative_path, sha256, stat):
            return False
        final = self.stamp_path(relative_path, sha256, stat)
        payload = dict(self._payload(relative_path, sha256, stat), verified_at=_utc_now())
        self.stamp_dir.mkdir(parents=True, exist_ok=True)
        tmp = final.with_name(f".{final.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, final)
        finally:
            if tmp.exists():
                tmp.unlink()
        return True


def verify_file_sha256(
    path: str | Path,
    *,
    relative_path: str,
    expected_sha256: str,
    cache: HashStampCache | None,
    force: bool = False,
    hasher: Callable[[Path], str] = sha256_file,
) -> HashCheck:
    """Verify ``path`` against ``expected_sha256``, hashing only when needed.

    ``force=True`` always hashes (explicit verify) and refreshes the stamp.
    Otherwise a valid stamp for the file's current (size, mtime_ns) skips the
    hash. A stamp is written only after a matching hash, and only if the file's
    stat did not change during hashing. Stamp write failures (read-only
    snapshot) never fail verification; they are reported in ``warning``.
    """
    target = Path(path)
    expected = str(expected_sha256).lower()
    before = FileStat.of(target)
    if not force and cache is not None and cache.lookup(relative_path, expected, before):
        return HashCheck(relative_path, expected, expected, METHOD_STAMP, True)
    actual = hasher(target)
    if actual != expected:
        return HashCheck(relative_path, expected, actual, METHOD_HASHED, False)
    written = False
    warning = None
    if cache is not None:
        after = FileStat.of(target)
        if after != before:
            warning = f"{relative_path}: file changed while hashing; no stamp written"
        else:
            try:
                written = cache.record(relative_path, actual, after)
            except OSError as exc:
                warning = f"{relative_path}: hash stamp not written ({exc.__class__.__name__}: {exc})"
    return HashCheck(relative_path, expected, actual, METHOD_HASHED, True, written, warning)


__all__ = [
    "FileStat",
    "HASH_STAMP_RELATIVE_DIR",
    "HASH_STAMP_SCHEMA_VERSION",
    "HashCheck",
    "HashStampCache",
    "METHOD_HASHED",
    "METHOD_SIZE_ONLY",
    "METHOD_STAMP",
    "sha256_file",
    "verify_file_sha256",
]
