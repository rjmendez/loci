"""Measured health of the local deployment, read from the LiteLLM cost-ledger snapshot.

The LiteLLM proxy's ops_logs collector writes ~/.loci/litellm_ledger_health.json every few minutes: per model
group, over the last hour, how many LOCAL attempts there were and what share succeeded with an answer (an
empty response counts as a failure). This module turns that snapshot into one yes/no for the router in
llm_local: "is local measurably failing right now?"

It is deliberately conservative. A missing, stale, thin or synthetic-only snapshot is NOT evidence of
degradation, so this can only ever move work away from local on recent, organic, measured failure. The
decision logic mirrors loci_litellm.ledger_summary.local_degraded in the LiteLLM plugin; keep them equal.

Env: LOCI_LEDGER_HEALTH_FILE (default ~/.loci/litellm_ledger_health.json),
     LOCI_LOCAL_HEALTH_MAX_AGE_S=1800, LOCI_LOCAL_HEALTH_MIN_N=20,
     LOCI_LOCAL_HEALTH_MIN_GOOD=0.9, LOCI_LOCAL_HEALTH_MAX_EMPTY=0.15
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional


def snapshot_path() -> str:
    return os.environ.get("LOCI_LEDGER_HEALTH_FILE") or str(Path.home() / ".loci" / "litellm_ledger_health.json")


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def read_snapshot(path: Optional[str] = None) -> Optional[dict]:
    try:
        with open(path or snapshot_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def check(snapshot: Optional[dict] = None, now: Optional[float] = None) -> tuple[bool, str, str]:
    """(degraded, reason, group). The group judged is the one with the most organic local attempts; if no
    group has organic traffic the answer is "not degraded" with that as the reason."""
    now = time.time() if now is None else now
    snap = snapshot if snapshot is not None else read_snapshot()
    if not isinstance(snap, dict) or snap.get("version") != 1:
        return False, "no usable health snapshot", ""
    max_age = _num("LOCI_LOCAL_HEALTH_MAX_AGE_S", 1800)
    age = now - float(snap.get("as_of") or 0)
    if age > max_age:
        return False, f"snapshot is {int(age)}s old (max {int(max_age)}s)", ""
    groups = {k: v for k, v in (snap.get("groups") or {}).items() if isinstance(v, dict)}
    organic = {k: v for k, v in groups.items() if v.get("basis") == "organic"}
    if not organic:
        return False, "no group has organic local traffic in the window", ""
    group = max(organic, key=lambda k: organic[k].get("n", 0))
    g = organic[group]
    min_n = int(_num("LOCI_LOCAL_HEALTH_MIN_N", 20))
    if g.get("n", 0) < min_n:
        return False, f"only {g.get('n', 0)} attempts for {group} (need {min_n})", group
    min_good, max_empty = _num("LOCI_LOCAL_HEALTH_MIN_GOOD", 0.9), _num("LOCI_LOCAL_HEALTH_MAX_EMPTY", 0.15)
    if g.get("good_rate", 1.0) < min_good:
        return True, f"{group}: good rate {g['good_rate']:.2f} below {min_good:.2f} over {g['n']} attempts", group
    if g.get("empty_rate", 0.0) > max_empty:
        return True, f"{group}: empty rate {g['empty_rate']:.2f} above {max_empty:.2f} over {g['n']} attempts", group
    return False, f"{group}: local meets the thresholds", group
