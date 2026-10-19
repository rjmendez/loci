"""A retrying fail-open latch: transient load failures back off and retry, real incompatibility sticks.

Lazy singletons (the reranker, the BM25 sparse embedder, the verdict backend) used to latch a
module global to ``False`` on ANY exception, so one memory-pressure ``OSError`` or one network blip
during a model load disabled the feature until the process was restarted, and restarting drops every
MCP client. ``_get_qdrant`` (60 s) and ``_get_ladybug`` (30 s) already retry; this is the shared
form of that pattern with exponential backoff and a state dict for ``loci_health``.

Usage (the caller keeps its own lock around the load so one thread loads while the rest wait)::

    if not latch.allow():          # permanent, or still inside the backoff window
        return None
    try:
        obj = load()
    except Exception as exc:
        latch.failure(exc)         # permanent only for ImportError (or permanent=True)
        return None
    latch.success()

Failure classes:
  * permanent  -- ``ImportError`` (module genuinely absent), or ``failure(exc, permanent=True)`` for an
    explicit incompatibility such as ``VerdictDimensionMismatch``. Never retried.
  * transient  -- everything else. Retried after ``base_s * factor**(failures-1)`` seconds, capped.

Every failure is logged at WARNING; repeats are rate-limited by ``log_every_s`` and report how many
were suppressed. The clock is injectable so tests drive the backoff exactly.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple, Type

logger = logging.getLogger("loci-mcp")

_ERROR_MAX_CHARS = 200


class RetryLatch:
    def __init__(self, name: str, base_s: float = 30.0, cap_s: float = 600.0, factor: float = 2.0,
                 permanent_types: Tuple[Type[BaseException], ...] = (ImportError,),
                 log_every_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time,
                 log: Optional[logging.Logger] = None):
        self.name = name
        self.base_s = float(base_s)
        self.cap_s = max(float(cap_s), self.base_s)
        self.factor = float(factor)
        self.permanent_types = tuple(permanent_types)
        self.log_every_s = float(log_every_s)
        self._clock = clock
        self._wall = wall
        self._log = log or logger
        self._lock = threading.Lock()
        self._failures_total = 0                 # cumulative: survives success() and reset()
        self._reset_locked()
        self._warned_at: dict = {}      # rate-limit key -> clock() of the last emitted warning
        self._suppressed: dict = {}     # rate-limit key -> warnings dropped since the last emitted one

    # ------------------------------------------------------------------ state
    def _reset_locked(self) -> None:
        self._permanent = False
        self._last_error = ""
        self._since: Optional[float] = None      # wall time of the first failure of the current run
        self._attempts = 0                       # consecutive failures
        self._retry_at: Optional[float] = None   # clock() when the next attempt is allowed

    def reset(self) -> None:
        """Forget the failure (the configuration changed, so the old failure says nothing about the new one)."""
        with self._lock:
            self._reset_locked()

    @property
    def permanent(self) -> bool:
        with self._lock:
            return self._permanent

    def delay_after(self, failures: int) -> float:
        """Backoff after the ``failures``-th consecutive failure: base * factor**(failures-1), capped."""
        if failures <= 0:
            return 0.0
        return min(self.cap_s, self.base_s * (self.factor ** (failures - 1)))

    def allow(self) -> bool:
        """May an attempt be made now? False when permanently failed or still inside the backoff window."""
        with self._lock:
            if self._permanent:
                return False
            return self._retry_at is None or self._clock() >= self._retry_at

    def success(self) -> None:
        """An attempt succeeded: clear the failure (last_error, attempts, backoff)."""
        with self._lock:
            recovered = self._attempts > 0
            attempts = self._attempts
            self._reset_locked()
        if recovered:
            self._log.warning("%s recovered after %d failed attempt(s)", self.name, attempts)

    def failure(self, exc: BaseException, permanent: Optional[bool] = None) -> bool:
        """Record a failed attempt and log it at WARNING (rate-limited). Returns True when it is permanent.

        ``permanent=None`` classifies by type (``permanent_types``); True/False overrides."""
        is_permanent = isinstance(exc, self.permanent_types) if permanent is None else bool(permanent)
        text = f"{type(exc).__name__}: {exc}"[:_ERROR_MAX_CHARS]
        with self._lock:
            self._attempts += 1
            self._failures_total += 1
            self._last_error = text
            if self._since is None:
                self._since = self._wall()
            if is_permanent:
                self._permanent = True
                self._retry_at = None
                delay = None
            else:
                delay = self.delay_after(self._attempts)
                self._retry_at = self._clock() + delay
            attempts = self._attempts
        if is_permanent:
            self.warn("failure", "%s unavailable permanently (attempt %d): %s", self.name, attempts, text)
        else:
            self.warn("failure", "%s failed (attempt %d), retrying in %.0fs: %s", self.name, attempts, delay, text)
        return is_permanent

    def state(self) -> dict:
        """Snapshot for loci_health. ``ok`` is False while a failure is outstanding."""
        with self._lock:
            if self._permanent or self._retry_at is None:
                nxt = None
            else:
                nxt = max(0.0, self._retry_at - self._clock())
            return {
                "name": self.name,
                "ok": self._attempts == 0 and not self._permanent,
                "permanent": self._permanent,
                "last_error": self._last_error,
                "since": (datetime.fromtimestamp(self._since, timezone.utc).isoformat(timespec="seconds")
                          if self._since is not None else ""),
                "attempts": self._attempts,
                "failures_total": self._failures_total,
                "next_retry_in_s": None if nxt is None else round(nxt, 3),
            }

    # ---------------------------------------------------------------- logging
    def warn(self, key: str, msg: str, *args) -> bool:
        """WARNING at most once per ``log_every_s`` for ``key``; returns whether it was emitted.
        The emitted line says how many repeats were suppressed since the previous one."""
        now = self._clock()
        with self._lock:
            last = self._warned_at.get(key)
            if last is not None and now - last < self.log_every_s:
                self._suppressed[key] = self._suppressed.get(key, 0) + 1
                return False
            dropped = self._suppressed.pop(key, 0)
            self._warned_at[key] = now
        if dropped:
            msg = msg + " (+%d similar suppressed)"
            args = args + (dropped,)
        self._log.warning(msg, *args)
        return True
