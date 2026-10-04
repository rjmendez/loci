"""Bounded, append-only JSONL for decision instrumentation.

Loci makes many automatable decisions (contradiction judging, memory surfacing,
routing) whose outcomes were never persisted, so there is nothing to learn or
audit them from. This module is the one writer those instrumentation logs share.

Rules every caller follows:

* Rows carry ids, scores, enums and counts only -- never finding text, query
  text, model output or reasons. A reader must be able to share an export of
  these files without sharing investigation content.
* Size is bounded. Before an append would take the live file past
  ``max_bytes`` the file is rotated: ``name`` -> ``name.1`` -> ... ->
  ``name.<keep>``; the generation past ``keep`` is dropped. The newest rows are
  always in ``name``. Worst-case disk use per log is about
  ``(keep + 1) * max_bytes``.
* It never raises. Instrumentation that fails must not fail the decision it
  records; ``append_rows`` returns False instead.

Defaults: 4 MiB per file and 3 rotated generations, overridable with
``LOCI_INSTRUMENTATION_LOG_MAX_BYTES`` and ``LOCI_INSTRUMENTATION_LOG_KEEP``.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 4 * 1024 * 1024
DEFAULT_KEEP = 3
_MIN_MAX_BYTES = 4096
# Non-blocking: one lock attempt, no wait. A contended instrumentation write is
# dropped (dropped rows are allowed) rather than delaying a hot path such as memory_surface.
_LOCK_TIMEOUT_S = 0.0


def env_enabled(name: str, default: bool) -> bool:
    """Read an on/off flag. Unset or blank means ``default``; "0/false/no/off" mean off."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return max(minimum, value)


def max_bytes() -> int:
    return _env_int("LOCI_INSTRUMENTATION_LOG_MAX_BYTES", DEFAULT_MAX_BYTES, _MIN_MAX_BYTES)


def keep_generations() -> int:
    return _env_int("LOCI_INSTRUMENTATION_LOG_KEEP", DEFAULT_KEEP, 1)


def _rotated(path: Path, generation: int) -> Path:
    return path.with_name(f"{path.name}.{generation}")


def _rotate(path: Path, keep: int) -> None:
    oldest = _rotated(path, keep)
    if oldest.exists():
        oldest.unlink()
    for generation in range(keep - 1, 0, -1):
        src = _rotated(path, generation)
        if src.exists():
            os.replace(src, _rotated(path, generation + 1))
    os.replace(path, _rotated(path, 1))


def append_rows(
    path: Path,
    rows: Iterable[dict],
    *,
    limit_bytes: int | None = None,
    keep: int | None = None,
) -> bool:
    """Append ``rows`` to ``path`` as JSONL, rotating first if the file would outgrow its cap.

    All rows of one call land in the same file generation. Writers serialise on a
    sidecar ``<name>.lock`` so a rotation cannot interleave with another append.
    Returns True when the rows were written (or there were none), False on any error.
    """
    try:
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        if not payload:
            return True
        cap = max_bytes() if limit_bytes is None else max(1, int(limit_bytes))
        generations = keep_generations() if keep is None else max(1, int(keep))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        from inv_store import _locked_file

        # No lock wait (_LOCK_TIMEOUT_S = 0): a contended instrumentation write is
        # dropped (False), never allowed to stall the store or search path it is recording.
        with _locked_file(path.with_name(path.name + ".lock"), "a+", exclusive=True,
                          timeout_s=_LOCK_TIMEOUT_S):
            size = path.stat().st_size if path.exists() else 0
            if size and size + len(payload.encode("utf-8")) > cap:
                _rotate(path, generations)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
        return True
    except Exception as exc:
        logger.debug("instrumentation append to %s failed (fail-open): %r", path, exc)
        return False
