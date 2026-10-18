#!/usr/bin/env python3
"""Policy watchdog core: decide swap-to-incumbent from policy rows. NOT SCHEDULED in this slice.

Tripwire-Governor S15. ``run_once`` is a pure decision core: it reads an iterator of policy rows, judges
every learned arm against the frozen incumbent with ``loci.degeneracy.judge_arm`` and reports through
injected sinks. The clock, the latch file and every sink are injected, so it touches no live store, calls
no model and makes no network call. Wiring it to a timer is a later, separate step.

Rules it enforces:

* A silent monitor is an alarm. ``rows_read == 0`` is a failure (``rows_read_zero``), as is a window in
  which every row was malformed (``all_rows_malformed``). The own, the governor's and the nightly
  heartbeat each alarm when older than ``max_silence`` seconds (default 900 = 15 min) or never seen.
* DEGENERATE swaps in the frozen incumbent and NEVER auto-resumes. The verdict is latched in a JSON state
  file; once latched, every later run still reports the swap (``swap_sink`` must be idempotent) even if
  the arm's recent rows look healthy. Only ``LatchStore.clear`` (operator + reason) lifts it.
* TOO_EARLY and HEALTHY do nothing.

Row format (one JSON object per line in the CLI): ``{"arm": str, "ts": epoch_seconds, "completed": bool,
"refused": bool, "escalated": bool, "cost_usd": number, "quality": number | null}``.

    python scripts/policy_watchdog.py --rows rows.jsonl --incumbent static --learned router-v1 \\
        --state latch.json --heartbeat-dir hb/ --alarm-log alarms.jsonl --swap-log swaps.jsonl

Exit status: 0 nothing to do, 2 at least one alarm, 3 a swap is in force (no alarm).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loci.degeneracy import (  # noqa: E402
    DEGENERATE, MIN_ARM_N, DegeneracyInputError, Verdict, judge_arm, summarize_arm,
)

__all__ = [
    "DEFAULT_MAX_SILENCE_S", "HEARTBEAT_NAMES", "WatchdogConfig", "Decision", "LatchStore", "run_once", "main",
]

DEFAULT_MAX_SILENCE_S = 900.0  # 15 min
HEARTBEAT_NAMES = ("watchdog", "governor", "nightly")


@dataclass(frozen=True)
class WatchdogConfig:
    incumbent: str
    learned: tuple[str, ...]
    max_silence: Mapping[str, float] = field(
        default_factory=lambda: {name: DEFAULT_MAX_SILENCE_S for name in HEARTBEAT_NAMES})
    window_s: float = 86400.0
    min_n: int = MIN_ARM_N


@dataclass(frozen=True)
class Decision:
    swap: tuple[str, ...]  # arms to serve from the incumbent (new verdicts and latched ones)
    verdicts: Mapping[str, Verdict]  # this run's verdict per learned arm judged (latched arms are not re-judged)
    alarms: tuple[str, ...]
    rows_read: int
    rows_bad: int


class LatchStore:
    """JSON file ``{"degenerate": {arm: {"at": ts, "reasons": [...]}}, "cleared": [...]}``."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"degenerate": {}, "cleared": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def latched(self) -> dict[str, dict[str, Any]]:
        return dict(self._load()["degenerate"])

    def latch(self, arm: str, at: float, reasons: Sequence[str]) -> None:
        state = self._load()
        state["degenerate"].setdefault(arm, {"at": at, "reasons": list(reasons)})
        self._write(state)

    def clear(self, arm: str, operator: str, reason: str, at: float) -> None:
        """The only way back: an explicit operator act, recorded."""
        if not operator.strip() or not reason.strip():
            raise ValueError("clearing a latch needs an operator and a reason")
        state = self._load()
        if arm not in state["degenerate"]:
            raise KeyError(f"arm {arm!r} is not latched")
        was = state["degenerate"].pop(arm)
        state["cleared"].append({"arm": arm, "operator": operator, "reason": reason, "at": at, "was": was})
        self._write(state)

    def _write(self, state: dict[str, Any]) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True, indent=1), encoding="utf-8")
        tmp.replace(self.path)


def _ingest(row: Any, now: float, window_s: float) -> Optional[tuple[str, Mapping]]:
    """(arm, row) for a well-formed in-window row; None for a malformed one; ('', row) for out-of-window."""
    if not isinstance(row, dict) or not isinstance(row.get("arm"), str):
        return None
    ts = row.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
        return None
    try:
        float(row["cost_usd"])
    except (KeyError, TypeError, ValueError):
        return None
    if ts < now - window_s:
        return ("", row)
    return (row["arm"], row)


def run_once(
    rows: Iterable[Any],
    *,
    config: WatchdogConfig,
    heartbeats: Mapping[str, Optional[float]],
    now: Callable[[], float],
    latch: LatchStore,
    swap_sink: Callable[[dict[str, Any]], None],
    alarm_sink: Callable[[dict[str, Any]], None],
    heartbeat_sink: Callable[[str, float], None],
) -> Decision:
    t = now()
    by_arm: dict[str, list[Mapping]] = {}
    rows_read = rows_bad = 0
    wanted = {config.incumbent, *config.learned}
    for raw in rows:
        rows_read += 1
        got = _ingest(raw, t, config.window_s)
        if got is None:
            rows_bad += 1
        elif got[0] in wanted:
            by_arm.setdefault(got[0], []).append(got[1])

    alarms: list[str] = []
    if rows_read == 0:
        alarms.append("rows_read_zero")
    elif rows_bad == rows_read:
        alarms.append("all_rows_malformed")
    for name, limit in config.max_silence.items():
        last = heartbeats.get(name)
        if last is None or t - last > limit:
            alarms.append(f"heartbeat_silent:{name}")

    already = latch.latched()
    verdicts: dict[str, Verdict] = {}
    newly: dict[str, Verdict] = {}
    pending = [arm for arm in config.learned if arm not in already]
    if pending:
        try:
            inc = summarize_arm(config.incumbent, by_arm.get(config.incumbent, ()))
            for arm in pending:
                v = judge_arm(summarize_arm(arm, by_arm.get(arm, ())), inc, min_n=config.min_n)
                verdicts[arm] = v
                if v.label == DEGENERATE:
                    newly[arm] = v
        except DegeneracyInputError as exc:
            alarms.append(f"unjudgeable:{exc}")

    for arm, v in newly.items():  # persist the latch BEFORE any sink runs
        latch.latch(arm, t, v.reasons)
    swap = tuple(arm for arm in config.learned if arm in already or arm in newly)
    for arm in swap:
        reasons = list(newly[arm].reasons) if arm in newly else list(already[arm]["reasons"])
        swap_sink({"arm": arm, "to": config.incumbent, "ts": t, "reasons": reasons, "new": arm in newly})
    for code in alarms:
        alarm_sink({"alarm": code, "ts": t})
    heartbeat_sink("watchdog", t)
    return Decision(swap=swap, verdicts=verdicts, alarms=tuple(alarms), rows_read=rows_read, rows_bad=rows_bad)


def _read_jsonl(path: Path) -> Iterable[Any]:
    if not path.exists():
        return
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError:
                yield line  # counted as a malformed row


def _append(path: Path) -> Callable[[dict[str, Any]], None]:
    def sink(entry: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return sink


def _read_heartbeats(directory: Path) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    for name in HEARTBEAT_NAMES:
        try:
            out[name] = float((directory / name).read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            out[name] = None
    return out


def main(argv: Optional[list[str]] = None, *, clock: Callable[[], float] = time.time) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", type=Path, required=True, help="policy rows, one JSON object per line")
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--learned", action="append", required=True, help="learned arm name (repeatable)")
    ap.add_argument("--state", type=Path, required=True, help="latch file")
    ap.add_argument("--heartbeat-dir", type=Path, required=True,
                    help="directory with files watchdog, governor, nightly holding the last epoch seconds")
    ap.add_argument("--alarm-log", type=Path, required=True)
    ap.add_argument("--swap-log", type=Path, required=True)
    ap.add_argument("--window-s", type=float, default=86400.0)
    ap.add_argument("--clear-latch", metavar="ARM", help="operator resume of a latched arm (needs --operator, --reason)")
    ap.add_argument("--operator", default="")
    ap.add_argument("--reason", default="")
    args = ap.parse_args(argv)

    latch = LatchStore(args.state)
    if args.clear_latch:
        latch.clear(args.clear_latch, args.operator, args.reason, clock())
        return 0
    cfg = WatchdogConfig(incumbent=args.incumbent, learned=tuple(args.learned), window_s=args.window_s)

    def beat(name: str, ts: float) -> None:
        (args.heartbeat_dir / name).write_text(repr(ts), encoding="utf-8")

    args.heartbeat_dir.mkdir(parents=True, exist_ok=True)
    decision = run_once(
        _read_jsonl(args.rows), config=cfg, heartbeats=_read_heartbeats(args.heartbeat_dir), now=clock,
        latch=latch, swap_sink=_append(args.swap_log), alarm_sink=_append(args.alarm_log), heartbeat_sink=beat,
    )
    print(json.dumps({"rows_read": decision.rows_read, "rows_bad": decision.rows_bad, "swap": list(decision.swap),
                      "alarms": list(decision.alarms),
                      "verdicts": {a: [v.label, list(v.reasons)] for a, v in decision.verdicts.items()}}))
    return 2 if decision.alarms else 3 if decision.swap else 0


if __name__ == "__main__":
    sys.exit(main())
