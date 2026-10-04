#!/usr/bin/env python3
"""Read-only report on the D10 shadow gate log (docs/d10_shadow_gate.md).

Reads <data home>/instrumentation/d10_shadow.jsonl (and its rotated generations) and,
for the investigations it names, <memory dir>/<investigation>/judge_verdicts.jsonl.
Prints aggregate counts as JSON to stdout. It opens every file read-only and writes
nothing anywhere: no file, no store, no index.

What it reports
  gates        cosine keep/drop vs MLP keep/drop over all logged pairs (2x2), and the
               same for "in the prompt" (each gate's top-12 of its kept pairs).
  latency      per-pair shadow scoring time (p50/p95/max, microseconds).
  judge        for judged finding pairs (new_finding_id, neighbor_id) whose two findings
               were both scored in one shadow call: whether each gate kept both, dropped
               both, or split them, by verdict.

How to read the judge join. The contradiction judge rules on finding-finding pairs
(agree / contradict / same_topic_no_conflict); it never rules on whether a finding is
relevant to the question investigation_reason was asked. Every judged pair is a
same-topic pair, so the join measures COHERENCE: a gate that splits a same-topic pair
keeps one half of a topic and drops the other. It is not an accuracy score, and no
verdict says which of the two findings should have been kept.

Usage:
  python3 scripts/d10_shadow_report.py [--shadow-log PATH] [--memory-dir DIR]
Defaults follow the server: LOCI_MEMORY_DIR, else ~/.loci/memory-sessions, else the
legacy ~/.hermes/memory-sessions; the log is <that dir's parent>/instrumentation/d10_shadow.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

JUDGED_VERDICTS = ("agree", "contradict", "same_topic_no_conflict")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_GENERATIONS = 32


def default_memory_dir() -> Path:
    """The server's resolution (mcp/legacy_env.py memory_dir), repeated so this stays stdlib-only."""
    explicit = os.environ.get("LOCI_MEMORY_DIR") or os.environ.get("HERMES_MEMORY_DIR")
    if explicit:
        return Path(explicit).expanduser()
    new = Path.home() / ".loci" / "memory-sessions"
    legacy = Path.home() / ".hermes" / "memory-sessions"
    return new if new.is_dir() or not legacy.is_dir() else legacy


def generations(path: Path) -> list[Path]:
    """The live file and its rotated generations (name, name.1, ...), oldest first."""
    found = [p for p in (path.with_name(f"{path.name}.{g}") for g in range(_MAX_GENERATIONS, 0, -1)) if p.is_file()]
    return found + ([path] if path.is_file() else [])


def read_jsonl(paths: Iterable[Path]) -> Iterator[dict]:
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    yield row


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def _two_by_two(rows: list[dict], a: str, b: str) -> dict:
    c = Counter((bool(r.get(a)), bool(r.get(b))) for r in rows)
    n = sum(c.values())
    agree = c[(True, True)] + c[(False, False)]
    return {"both_keep": c[(True, True)], "both_drop": c[(False, False)],
            "cosine_only_keep": c[(True, False)], "mlp_only_keep": c[(False, True)],
            "agreement": round(agree / n, 4) if n else None}


def report(shadow_log: Path, memory_dir: Path) -> dict:
    rows = [r for r in read_jsonl(generations(shadow_log)) if r.get("event") == "d10_shadow"]
    calls: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        calls[str(r.get("call_id"))].append(r)
    invs = sorted({str(r.get("investigation_id")) for r in rows})
    per_pair = [float(r["latency_us_per_pair"]) for r in rows
                if isinstance(r.get("latency_us_per_pair"), (int, float))]
    out: dict = {
        "shadow_log": str(shadow_log),
        "n_rows": len(rows),
        "n_calls": len(calls),
        "n_investigations": len(invs),
        "gate_versions": dict(Counter(str(r.get("gate_version")) for r in rows)),
        "calls_truncated": sum(1 for c in calls.values() if c and c[0].get("n_logged", 0) < c[0].get("n_pairs", 0)),
        "shadow_errors_total_max": max((int(r.get("shadow_errors_total") or 0) for r in rows), default=0),
        "keep_vs_keep": _two_by_two(rows, "cos_keep", "mlp_keep"),
        "prompt_vs_prompt": _two_by_two(rows, "cos_in_context", "mlp_in_context"),
        "latency_us_per_pair": {"p50": _pct(per_pair, 0.5), "p95": _pct(per_pair, 0.95),
                                "max": max(per_pair) if per_pair else None},
    }

    by_verdict: dict[str, Counter] = defaultdict(Counter)
    skipped_ids = 0
    judged_rows = 0
    for inv in invs:
        if not _SAFE_ID.match(inv):
            skipped_ids += 1
            continue
        decisions = [{str(r.get("finding_id")): r for r in c} for c in calls.values()
                     if c and str(c[0].get("investigation_id")) == inv]
        for v in read_jsonl(generations(memory_dir / inv / "judge_verdicts.jsonl")):
            judged_rows += 1
            verdict = v.get("verdict") if v.get("judge_ok") else None
            key = str(verdict) if verdict in JUDGED_VERDICTS else "no_verdict"
            a, b = str(v.get("new_finding_id")), str(v.get("neighbor_id"))
            joined = [(d[a], d[b]) for d in decisions if a in d and b in d]
            if not joined:
                by_verdict[key]["not_in_one_shadow_call"] += 1
                continue
            for ra, rb in joined:
                by_verdict[key]["joined"] += 1
                for gate, field in (("cosine", "cos_keep"), ("mlp", "mlp_keep")):
                    ka, kb = bool(ra.get(field)), bool(rb.get(field))
                    by_verdict[key][f"{gate}_{'both_keep' if ka and kb else 'both_drop' if not (ka or kb) else 'split'}"] += 1
    judge = {"verdict_rows_read": judged_rows, "investigation_ids_skipped_as_unsafe": skipped_ids,
             "by_verdict": {k: dict(v) for k, v in sorted(by_verdict.items())}}
    for gate in ("cosine", "mlp"):
        n = sum(v.get("joined", 0) for k, v in by_verdict.items() if k in JUDGED_VERDICTS)
        coherent = sum(v.get(f"{gate}_both_keep", 0) + v.get(f"{gate}_both_drop", 0)
                       for k, v in by_verdict.items() if k in JUDGED_VERDICTS)
        judge[f"{gate}_coherence_on_judged_pairs"] = round(coherent / n, 4) if n else None
    out["judge"] = judge
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--memory-dir", type=Path, default=None)
    ap.add_argument("--shadow-log", type=Path, default=None)
    args = ap.parse_args(argv)
    memory_dir = args.memory_dir or default_memory_dir()
    shadow_log = args.shadow_log or memory_dir.parent / "instrumentation" / "d10_shadow.jsonl"
    json.dump(report(shadow_log, memory_dir), sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
