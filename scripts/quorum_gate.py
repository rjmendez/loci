#!/usr/bin/env python3
"""
Quorum-sensing trigger gate for hermes_memory.

Biological analog: bacterial quorum sensing — actions only fire when enough
correlated signal accumulates above a threshold, preventing per-event chatter
and enabling coherent batch-level responses.

Each topic cluster maintains a decaying accumulator. Events deposit signal;
exponential decay erodes it over time. When the accumulator crosses a
threshold, the caller's action fires and the accumulator resets.

Usage (any script or cron):

    from quorum_gate import QuorumGate

    gate = QuorumGate()

    # At write time — deposit signal for a topic:
    gate.deposit("uds_inversion", amount=1.0)

    # Before an expensive collective action — check quorum:
    if gate.check_quorum("uds_inversion", threshold=5.0):
        run_consolidation_for_topic("uds_inversion")
        gate.reset("uds_inversion")

State is persisted in JSON so decay survives process restarts.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Optional

_HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
_DEFAULT_STATE_PATH = os.path.join(_HERMES_HOME, "quorum-state.json")

# How fast signal decays. Halves every HALFLIFE_SECONDS. Default: 30 min.
DEFAULT_HALFLIFE_SECONDS = float(
    os.environ.get("QUORUM_HALFLIFE_SECONDS", str(30 * 60))
)
_LN2 = math.log(2)


class QuorumGate:
    """Persistent decaying accumulator per topic cluster."""

    def __init__(self, state_path: Optional[str] = None,
                 halflife_seconds: float = DEFAULT_HALFLIFE_SECONDS):
        self._path = Path(state_path or _DEFAULT_STATE_PATH)
        self._halflife = halflife_seconds
        self._state: dict = self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(self._path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._state, f)
        os.replace(tmp, self._path)

    # ── decay ────────────────────────────────────────────────────────────────

    def _effective(self, topic: str) -> float:
        """Current accumulator value after exponential decay."""
        entry = self._state.get(topic)
        if not entry:
            return 0.0
        stored_value: float = entry["value"]
        stored_ts: float    = entry["ts"]
        age = max(0.0, time.time() - stored_ts)
        return stored_value * math.exp(-_LN2 * age / self._halflife)

    # ── public API ───────────────────────────────────────────────────────────

    def deposit(self, topic: str, amount: float = 1.0) -> float:
        """Add signal to a topic's accumulator. Returns new effective value."""
        current = self._effective(topic)
        new_value = current + amount
        self._state[topic] = {"value": new_value, "ts": time.time()}
        self._save()
        return new_value

    def effective(self, topic: str) -> float:
        """Read the current (decayed) accumulator value without modifying it."""
        return self._effective(topic)

    def check_quorum(self, topic: str, threshold: float) -> bool:
        """True if the topic's accumulator has reached the threshold."""
        return self._effective(topic) >= threshold

    def reset(self, topic: str) -> None:
        """Reset a topic's accumulator to zero (call after firing the action)."""
        self._state.pop(topic, None)
        self._save()

    def snapshot(self) -> dict[str, float]:
        """Return {topic: effective_value} for all tracked topics."""
        return {t: self._effective(t) for t in self._state}

    def prune_cold(self, floor: float = 0.01) -> list[str]:
        """Remove topics whose effective value has decayed below floor."""
        cold = [t for t in list(self._state) if self._effective(t) < floor]
        for t in cold:
            self._state.pop(t)
        if cold:
            self._save()
        return cold


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    """
    quorum_gate.py <cmd> [args]

    Commands:
      deposit <topic> [amount]         — add signal (default amount=1.0)
      check   <topic> <threshold>      — exit 0 if quorum reached, 1 if not
      reset   <topic>                  — zero the accumulator
      status                           — print all accumulators
      prune                            — remove decayed-out topics
    """
    import sys
    args = sys.argv[1:]
    gate = QuorumGate()

    if not args:
        print(_cli.__doc__)
        return

    cmd, *rest = args
    if cmd == "deposit":
        topic = rest[0]
        amount = float(rest[1]) if len(rest) > 1 else 1.0
        val = gate.deposit(topic, amount)
        print(f"{topic}: {val:.3f}")
    elif cmd == "check":
        topic, threshold = rest[0], float(rest[1])
        reached = gate.check_quorum(topic, threshold)
        print(f"{topic}: {gate.effective(topic):.3f} {'≥' if reached else '<'} {threshold}")
        sys.exit(0 if reached else 1)
    elif cmd == "reset":
        gate.reset(rest[0])
        print(f"reset: {rest[0]}")
    elif cmd == "status":
        snap = gate.snapshot()
        if not snap:
            print("(no active topics)")
        else:
            for topic, val in sorted(snap.items(), key=lambda x: -x[1]):
                print(f"  {topic:40s} {val:.3f}")
    elif cmd == "prune":
        removed = gate.prune_cold()
        print(f"pruned {len(removed)}: {removed}")
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
