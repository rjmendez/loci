"""Minimal fcntl compatibility shim for Windows.

This exposes the subset of the Unix API used by the repo: advisory-lock
constants and ``flock``. On Windows it is backed by ``msvcrt.locking`` on the
first byte of the file; with no ``msvcrt`` it is a no-op.

Semantics worth knowing:

* ``LOCK_UN`` releases. (An earlier version masked the operation with ``& 0x7``
  before comparing to ``LOCK_UN`` (8), so an unlock never matched, fell through to
  a blocking ``LK_LOCK`` and burned ~10 s retrying against the lock the caller
  itself held, then released nothing until the fd was closed.)
* The lock and its release both act on byte 0. ``msvcrt.locking`` works at the
  current file position, so the position is moved to 0 for the call and put back,
  otherwise a lock taken before a write could not be released after it.
* ``LOCK_SH`` is treated as exclusive: Windows byte-range locks have no shared mode here.
* ``LOCK_NB`` failures raise ``OSError`` so the callers' retry loops work. A blocking
  request (no ``LOCK_NB``) is retried by ``msvcrt`` for about 10 s and then fails open.
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
    """Best-effort advisory lock. See the module docstring for the semantics."""
    if msvcrt is None:
        return
    pos = os.lseek(fd, 0, os.SEEK_CUR)
    os.lseek(fd, 0, os.SEEK_SET)
    try:
        if operation & LOCK_UN:
            try:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass  # nothing held at byte 0 (never locked, or already released)
            return
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK if operation & LOCK_NB else msvcrt.LK_LOCK, 1)
        except OSError:
            if operation & LOCK_NB:
                raise
    finally:
        os.lseek(fd, pos, os.SEEK_SET)


__all__ = ["LOCK_SH", "LOCK_EX", "LOCK_NB", "LOCK_UN", "flock"]
