"""Registration records committed by hash, and the held-out-scored-once ledger (design.md section 6).

A registration is a dict of the choices a learned-router comparison is not allowed to make after seeing
data: features, estimator, margin, cluster unit, MDE table per cell, decay constants (from ONE named
source), split fingerprint, pass rule. ``PreregStore.register`` writes its canonical JSON exactly once and
returns the sha256; a second write with a different body for the same registration id refuses, the same
body is an idempotent no-op. ``ScoredOnceLedger`` makes the held-out rows score once per
(policy family, split fingerprint); scoring again needs an amendment id, and each amendment id is single
use.

Offline, stdlib only. The clock is injected.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from .stats import required_n_per_arm

__all__ = [
    "RegistrationError", "RegistrationConflict", "RegistrationTampered", "ScoredTwiceError",
    "LedgerCorruptError", "REQUIRED_FIELDS", "canonical_json", "record_hash", "split_fingerprint",
    "build_mde_table", "validate_registration", "PreregStore", "ScoredOnceLedger",
]

REQUIRED_FIELDS = ("features", "estimator", "margin", "cluster_unit", "mde_table", "decay_constants",
                   "split_fingerprint", "pass_rule")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class RegistrationError(ValueError):
    """A registration record or id is not acceptable."""


class RegistrationConflict(RegistrationError):
    """The registration id already exists with a different body."""


class RegistrationTampered(RegistrationError):
    """The stored file no longer hashes to the expected sha256."""


class ScoredTwiceError(RuntimeError):
    """The held-out split was already scored for this policy family; an amendment id is needed."""


class LedgerCorruptError(RuntimeError):
    """The scored-once ledger has a line that cannot be parsed; refuse rather than guess."""


def canonical_json(obj: Any) -> bytes:
    """Deterministic UTF-8 JSON: sorted keys, no whitespace, NaN/inf rejected."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except ValueError as exc:
        raise RegistrationError(f"record is not canonical-JSON serialisable: {exc}") from exc


def record_hash(record: Any) -> str:
    return hashlib.sha256(canonical_json(record)).hexdigest()


def split_fingerprint(row_ids: Iterable[str]) -> str:
    """sha256 over the sorted, de-duplicated row ids joined by newlines: order-independent."""
    body = "\n".join(sorted({str(r) for r in row_ids}))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def build_mde_table(cells: Mapping[str, float], margin: float, alpha: float = 0.05,
                    power: float = 0.8) -> dict[str, dict[str, Any]]:
    """``{cell: baseline success rate p}`` -> ``{cell: {"p": p, "n_per_arm": n}}`` for margin (positive)."""
    return {cell: {"p": p, "n_per_arm": required_n_per_arm(p, margin, alpha, power)}
            for cell, p in sorted(cells.items())}


def validate_registration(record: Mapping[str, Any]) -> None:
    """Raise ``RegistrationError`` naming the first thing wrong; return None when the record is complete."""
    if not isinstance(record, Mapping):
        raise RegistrationError("registration must be a mapping")
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise RegistrationError(f"registration missing fields: {', '.join(missing)}")
    margin = record["margin"]
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin >= 0:
        raise RegistrationError(f"margin must be a negative number (a tolerated drop), got {margin!r}")
    for name in ("features", "mde_table", "decay_constants", "pass_rule"):
        if not record[name]:
            raise RegistrationError(f"{name} must not be empty")
    for name in ("estimator", "cluster_unit", "split_fingerprint"):
        if not isinstance(record[name], str) or not record[name].strip():
            raise RegistrationError(f"{name} must be a non-empty string")
    for cell, row in record["mde_table"].items():
        n = row.get("n_per_arm") if isinstance(row, Mapping) else None
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise RegistrationError(f"mde_table[{cell!r}] needs a positive integer n_per_arm, got {n!r}")
    source = record["decay_constants"].get("source") if isinstance(record["decay_constants"], Mapping) else None
    if not isinstance(source, str) or not source.strip():
        raise RegistrationError("decay_constants must name its single 'source'")
    canonical_json(record)  # must serialise


def _check_id(reg_id: str) -> str:
    if not isinstance(reg_id, str) or not _ID_RE.fullmatch(reg_id):
        raise RegistrationError(f"bad registration id {reg_id!r}: use letters, digits, '.', '_' or '-'")
    return reg_id


class PreregStore:
    """A directory of write-once ``<id>.json`` files holding canonical JSON."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, reg_id: str) -> Path:
        return self.root / f"{_check_id(reg_id)}.json"

    def register(self, reg_id: str, record: Mapping[str, Any]) -> str:
        """Commit ``record`` under ``reg_id``; returns its sha256. Same body again returns the same hash;
        a different body raises ``RegistrationConflict`` and leaves the stored file untouched."""
        path = self.path(reg_id)
        validate_registration(record)
        body = canonical_json(record)
        digest = hashlib.sha256(body).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "xb") as fh:  # exclusive create: never overwrites
                fh.write(body)
                fh.flush()
                os.fsync(fh.fileno())
        except FileExistsError:
            existing = hashlib.sha256(path.read_bytes()).hexdigest()
            if existing != digest:
                raise RegistrationConflict(
                    f"registration {reg_id!r} already exists with a different body "
                    f"(stored {existing[:12]}, offered {digest[:12]}); register a new id") from None
        return digest

    def verify(self, reg_id: str, expected_sha256: str) -> dict[str, Any]:
        """Re-hash the stored bytes and return the record, or raise ``RegistrationTampered``."""
        data = self.path(reg_id).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_sha256:
            raise RegistrationTampered(
                f"registration {reg_id!r} hashes to {actual[:12]}, expected {expected_sha256[:12]}")
        return json.loads(data.decode("utf-8"))


class ScoredOnceLedger:
    """Append-only JSONL: one line per scoring of a (policy family, split fingerprint)."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self._clock = clock

    def _entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                entry["policy_family"], entry["split_fingerprint"]  # noqa: B018 -- shape check
            except (ValueError, KeyError, TypeError) as exc:
                raise LedgerCorruptError(f"{self.path}:{lineno}: unreadable ledger line") from exc
            out.append(entry)
        return out

    def times_scored(self, policy_family: str, fingerprint: str) -> int:
        return sum(1 for e in self._entries()
                   if e["policy_family"] == policy_family and e["split_fingerprint"] == fingerprint)

    def record_scoring(self, policy_family: str, fingerprint: str, *, amendment_id: Optional[str] = None) -> int:
        """Record that the held-out split is being scored; returns how many times it has now been scored.

        The first scoring needs nothing. Any later one raises ``ScoredTwiceError`` unless an
        ``amendment_id`` that has never been used is given.
        """
        entries = self._entries()
        prior = [e for e in entries
                 if e["policy_family"] == policy_family and e["split_fingerprint"] == fingerprint]
        if prior:
            if not amendment_id or not str(amendment_id).strip():
                raise ScoredTwiceError(
                    f"held-out split {fingerprint[:12]} already scored for {policy_family!r}; "
                    f"scoring again needs an amendment id")
            if any(e.get("amendment_id") == amendment_id for e in entries):
                raise ScoredTwiceError(f"amendment id {amendment_id!r} was already used; an amendment is single use")
        line = json.dumps({"policy_family": policy_family, "split_fingerprint": fingerprint,
                           "amendment_id": amendment_id, "scored_at": self._clock()}, sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return len(prior) + 1
