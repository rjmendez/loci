"""Minimal fcntl compatibility shim for Windows test environments.

This exposes the subset of the Unix API used by the repo: advisory-lock
constants and ``flock``. The implementation is intentionally tiny and fail-open
for the local test harness: it never blocks and always returns success for a
lock acquisition attempt. This is sufficient for the Loci JSONL store tests on
Windows where the Unix ``fcntl`` module is unavailable.
"""

from __future__ import annotations

import os

try:  # pragma: no cover - only on Windows
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover
    msvcrt = None  # type: ignore

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8


def flock(fd: int, operation: int) -> None:
    """Best-effort lock shim. The repo only uses this to gate file writes.

    On Windows, ``msvcrt.locking`` handles the advisory lock when available.
    For compatibility with the repository's retry logic, a non-blocking lock
    failure raises ``OSError`` so the caller can retry. Otherwise the call is a
    no-op.
    """
    if msvcrt is None:
        return
    mode = operation & 0x7
    try:
        if mode == LOCK_UN:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            return
        if operation & LOCK_NB:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
    except OSError:
        if operation & LOCK_NB:
            raise


__all__ = ["LOCK_SH", "LOCK_EX", "LOCK_NB", "LOCK_UN", "flock"]
