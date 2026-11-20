"""Measure the local judge against gold and choose which verdicts the cloud teacher (role ``judge``) should re-judge.

Pure and offline: no model call, no store access. Positive class = "the answer is correct".

* ``calibrate(judge_rows, gold_rows)``: agreement, Cohen's kappa, precision, recall, leniency bias (false-accept rate
  = judged correct among gold-incorrect), a per-judge-model breakdown, and bootstrap confidence intervals with a
  caller-fixed seed. Rows may be dicts or ``(item, value)`` pairs; see ``normalize_gold`` / ``normalize_judge``.
* ``select_escalations(...)``: disagreements with a second cheap signal always go up, then low-margin verdicts, then a
  seeded random audit fraction, all under a per-run call cap.
* ``recommend_threshold(...)``: smallest margin cut that catches a target share of the judge's errors.

CLI: ``python mcp/judge_calibration.py report --judge judge.jsonl --gold gold.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality_validators as qv  # noqa: E402

MAX_GRADE = 3
DEFAULT_THRESHOLD = 2
DEFAULT_BOOT = 1000
CI_LEVEL = 0.95
METRICS = ("agreement", "kappa", "precision", "recall", "leniency_bias")
_GOLD_KEYS = ("gold_correct", "correct", "grade", "relevance", "gold_grade")
_JUDGE_KEYS = ("verdict", "judge_accept", "correct")
_ID_KEYS = ("id", "item", "decision_id")


class CalibrationError(ValueError):
    """Malformed gold or judge input, or a bad parameter."""


# ------------------------------------------------------------------ adapters
def _id_of(row: Mapping) -> str:
    for k in _ID_KEYS:
        if row.get(k) not in (None, ""):
            return str(row[k])
    raise CalibrationError(f"row has no id (looked for {', '.join(_ID_KEYS)}): {dict(row)!r}")


def _check_threshold(threshold: int) -> int:
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 1 <= threshold <= MAX_GRADE:
        raise CalibrationError(f"threshold must be an int in 1..{MAX_GRADE}, got {threshold!r}")
    return threshold


def _gold_bool(value: Any, threshold: int, where: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and 0 <= value <= MAX_GRADE:
        return value >= threshold
    raise CalibrationError(f"{where}: gold value must be a bool or a grade 0..{MAX_GRADE}, got {value!r}")


def normalize_gold(rows: Iterable[Any], threshold: int = DEFAULT_THRESHOLD) -> Dict[str, bool]:
    """``{item id: gold_correct}``. A row is ``(item, bool|grade)`` or a dict with an id and one of
    ``gold_correct`` (bool) / ``grade`` / ``relevance`` / ``gold_grade`` (int 0..3, correct when >= threshold).
    A null ``gold_correct`` (an unlabelled candidate) is skipped. A repeated id is an error."""
    threshold = _check_threshold(threshold)
    out: Dict[str, bool] = {}
    for row in rows:
        if isinstance(row, Mapping):
            key = next((k for k in _GOLD_KEYS if k in row), None)
            if key is None:
                raise CalibrationError(f"gold row has none of {', '.join(_GOLD_KEYS)}: {dict(row)!r}")
            if row[key] is None:
                continue
            item, value = _id_of(row), row[key]
        else:
            try:
                item, value = row
            except (TypeError, ValueError):
                raise CalibrationError(f"gold row must be an (item, value) pair or a mapping, got {row!r}") from None
            item = str(item)
        if item in out:
            raise CalibrationError(f"gold item {item!r} appears twice")
        out[item] = _gold_bool(value, threshold, f"gold item {item!r}")
    return out


@dataclass(frozen=True)
class Verdict:
    item: str
    verdict: bool
    model: str = ""
    margin: Optional[float] = None
    signal: Optional[bool] = None


def _opt_bool(v: Any, name: str, item: str) -> Optional[bool]:
    if v is None or isinstance(v, bool):
        return v
    raise CalibrationError(f"judge item {item!r}: {name} must be a bool or null, got {v!r}")


def _margin(v: Any, item: str) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise CalibrationError(f"judge item {item!r}: margin must be a finite number or null, got {v!r}")
    return float(v)


def normalize_judge(rows: Iterable[Any]) -> Tuple[List[Verdict], int]:
    """(verdicts, n_unusable). A row is ``(item, bool)`` or a dict with an id and ``verdict`` / ``judge_accept`` /
    ``correct``, optional ``model`` / ``judge_model``, ``margin`` or ``confidence`` (higher = surer) and ``signal``
    (a second cheap verdict, e.g. the validators). A null verdict is unusable and counted. A repeated item keeps its
    LAST row (a re-judged answer counts once)."""
    latest: Dict[str, Verdict] = {}
    unusable: set = set()
    for row in rows:
        if isinstance(row, Mapping):
            item = _id_of(row)
            raw = next((row[k] for k in _JUDGE_KEYS if k in row), None)
            model = str(row.get("model") or row.get("judge_model") or "")
            margin = _margin(row.get("margin", row.get("confidence")), item)
            signal = _opt_bool(row.get("signal"), "signal", item)
        else:
            try:
                item, raw = row
            except (TypeError, ValueError):
                raise CalibrationError(f"judge row must be an (item, verdict) pair or a mapping, got {row!r}") from None
            item, model, margin, signal = str(item), "", None, None
        if raw is None:
            unusable.add(item)
            latest.pop(item, None)
            continue
        if not isinstance(raw, bool):
            raise CalibrationError(f"judge item {item!r}: verdict must be a bool or null, got {raw!r}")
        unusable.discard(item)
        latest[item] = Verdict(item, raw, model, margin, signal)
    return list(latest.values()), len(unusable)


# ------------------------------------------------------------------ metrics
def _div(a: int, b: int) -> Optional[float]:
    return a / b if b else None


def _metrics(pairs: Sequence[Tuple[bool, bool]]) -> Dict[str, Optional[float]]:
    """``pairs`` = (judge, gold). Undefined ratios are None, never 0."""
    tp = sum(j and g for j, g in pairs)
    fp = sum(j and not g for j, g in pairs)
    fn = sum(g and not j for j, g in pairs)
    tn = sum(not j and not g for j, g in pairs)
    kappa = qv.cohens_kappa(tp, fp, fn, tn)
    return {"agreement": _div(tp + tn, len(pairs)), "kappa": None if kappa is None else float(kappa),
            "precision": _div(tp, tp + fp), "recall": _div(tp, tp + fn), "leniency_bias": _div(fp, fp + tn)}


def _counts(pairs: Sequence[Tuple[bool, bool]]) -> Dict[str, int]:
    return {"n": len(pairs), "tp": sum(j and g for j, g in pairs), "fp": sum(j and not g for j, g in pairs),
            "fn": sum(g and not j for j, g in pairs), "tn": sum(not j and not g for j, g in pairs)}


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def bootstrap_ci(pairs: Sequence[Tuple[bool, bool]], seed: int, n_boot: int = DEFAULT_BOOT,
                 level: float = CI_LEVEL) -> Dict[str, Optional[Tuple[float, float]]]:
    """Percentile bootstrap CI per metric. Resamples that leave a metric undefined are dropped for that metric;
    None when none of them defined it. Same seed, same input => same interval."""
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise CalibrationError(f"n_boot must be a positive int, got {n_boot!r}")
    if not 0 < level < 1:
        raise CalibrationError(f"level must be in (0, 1), got {level!r}")
    rng = random.Random(seed)
    n = len(pairs)
    samples: Dict[str, List[float]] = {m: [] for m in METRICS}
    if n:
        for _ in range(n_boot):
            res = _metrics([pairs[rng.randrange(n)] for _ in range(n)])
            for m in METRICS:
                if res[m] is not None:
                    samples[m].append(res[m])
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    out: Dict[str, Optional[Tuple[float, float]]] = {}
    for m, vals in samples.items():
        vals.sort()
        out[m] = (_percentile(vals, lo_q), _percentile(vals, hi_q)) if vals else None
    return out


def _block(pairs: Sequence[Tuple[bool, bool]], seed: int, n_boot: int) -> Dict[str, Any]:
    return {**_counts(pairs), **_metrics(pairs), "ci": bootstrap_ci(pairs, seed, n_boot)}


def calibrate(judge_rows: Iterable[Any], gold_rows: Iterable[Any], *, threshold: int = DEFAULT_THRESHOLD,
              seed: int = 0, n_boot: int = DEFAULT_BOOT) -> Dict[str, Any]:
    """Judge vs gold over the items both have. Returns the overall block, ``by_model`` (judge model -> block; rows with
    no model are under ``""``), and the join accounting: ``n_unusable`` (null verdicts), ``n_no_gold`` (judged but not
    in gold) and ``n_not_judged`` (gold items the judge never rated)."""
    gold = normalize_gold(gold_rows, threshold)
    verdicts, unusable = normalize_judge(judge_rows)
    joined = [(v, gold[v.item]) for v in verdicts if v.item in gold]
    by_model: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
    for v, g in joined:
        by_model[v.model].append((v.verdict, g))
    out = _block([(v.verdict, g) for v, g in joined], seed, n_boot)
    out.update({"by_model": {m: _block(p, seed, n_boot) for m, p in sorted(by_model.items())},
                "n_unusable": unusable, "n_no_gold": sum(v.item not in gold for v in verdicts),
                "n_not_judged": len(set(gold) - {v.item for v in verdicts}), "threshold": threshold, "seed": seed,
                "n_boot": n_boot})
    return out


# ------------------------------------------------------------------ escalation
@dataclass(frozen=True)
class Escalation:
    item: str
    reason: str          # "disagreement" | "low_margin" | "audit"


def select_escalations(verdicts: Sequence[Verdict], *, call_cap: int, audit_fraction: float = 0.0,
                       margin_threshold: Optional[float] = None, seed: int = 0) -> List[Escalation]:
    """Which local verdicts the teacher judge re-judges, at most ``call_cap``.

    Priority: (1) the judge's verdict differs from its ``signal`` (lowest margin first, item id breaks ties),
    (2) ``margin <= margin_threshold``, (3) a random audit: each remaining item is drawn with probability
    ``audit_fraction`` by a ``seed``-ed generator over items sorted by id, so the draw does not depend on input order.
    The cap cuts the list from the end, so the audit is the first thing a tight cap drops."""
    if isinstance(call_cap, bool) or not isinstance(call_cap, int) or call_cap < 0:
        raise CalibrationError(f"call_cap must be a non-negative int, got {call_cap!r}")
    if isinstance(audit_fraction, bool) or not isinstance(audit_fraction, (int, float)) or not 0 <= audit_fraction <= 1:
        raise CalibrationError(f"audit_fraction must be in [0, 1], got {audit_fraction!r}")
    if margin_threshold is not None and (isinstance(margin_threshold, bool)
                                         or not isinstance(margin_threshold, (int, float))
                                         or not math.isfinite(margin_threshold)):
        raise CalibrationError(f"margin_threshold must be a finite number or None, got {margin_threshold!r}")
    ordered = sorted(verdicts, key=lambda v: v.item)

    def low_first(v: Verdict) -> Tuple[float, str]:
        return (math.inf if v.margin is None else v.margin, v.item)

    chosen: List[Escalation] = []
    taken: set = set()
    dis = sorted((v for v in ordered if v.signal is not None and v.signal != v.verdict), key=low_first)
    chosen += [Escalation(v.item, "disagreement") for v in dis]
    taken.update(v.item for v in dis)
    if margin_threshold is not None:
        low = sorted((v for v in ordered if v.item not in taken and v.margin is not None
                      and v.margin <= margin_threshold), key=low_first)
        chosen += [Escalation(v.item, "low_margin") for v in low]
        taken.update(v.item for v in low)
    rng = random.Random(seed)
    chosen += [Escalation(v.item, "audit") for v in ordered if v.item not in taken and rng.random() < audit_fraction]
    return chosen[:call_cap]


def recommend_threshold(verdicts: Sequence[Verdict], gold: Mapping[str, bool],
                        target_capture: float = 0.8) -> Optional[Dict[str, Any]]:
    """Smallest margin cut t such that escalating ``margin <= t`` catches ``target_capture`` of the judge's errors
    (verdict != gold) among items with both a margin and gold. None when there are no such errors."""
    if isinstance(target_capture, bool) or not isinstance(target_capture, (int, float)) or not 0 < target_capture <= 1:
        raise CalibrationError(f"target_capture must be in (0, 1], got {target_capture!r}")
    rows = [(v.margin, v.verdict != gold[v.item]) for v in verdicts if v.margin is not None and v.item in gold]
    errors = sum(e for _, e in rows)
    if not errors:
        return None
    need = math.ceil(target_capture * errors)
    for t in sorted({m for m, _ in rows}):
        caught = sum(e for m, e in rows if m <= t)
        if caught >= need:
            esc = sum(m <= t for m, _ in rows)
            return {"threshold": t, "errors": errors, "caught": caught, "error_capture": caught / errors,
                    "escalated": esc, "escalate_fraction": esc / len(rows)}
    raise AssertionError("unreachable: the largest margin catches every error")


# ------------------------------------------------------------------ CLI
def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CalibrationError(f"{path}:{n}: not JSON ({exc.msg})") from None
            if not isinstance(row, dict):
                raise CalibrationError(f"{path}:{n}: row is not an object")
            rows.append(row)
    return rows


def _f(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _line(label: str, b: Mapping) -> str:
    cells = []
    for m in METRICS:
        ci = b["ci"][m]
        cells.append(f"{m} {_f(b[m])}" + ("" if ci is None else f" [{ci[0]:.3f},{ci[1]:.3f}]"))
    return f"{label} n={b['n']} tp={b['tp']} fp={b['fp']} fn={b['fn']} tn={b['tn']}\n    " + "\n    ".join(cells)


def render_report(cal: Mapping, rec: Optional[Mapping], target: float) -> str:
    lines = [f"judge calibration (gold threshold {cal['threshold']}, seed {cal['seed']}, boot {cal['n_boot']})",
             f"unusable verdicts {cal['n_unusable']}, judged without gold {cal['n_no_gold']}, "
             f"gold not judged {cal['n_not_judged']}", _line("overall", cal)]
    lines += [_line(f"model {m or '(unnamed)'}", b) for m, b in cal["by_model"].items()]
    if rec is None:
        lines.append("recommended escalation threshold: none (no judge errors with a margin to learn from)")
    else:
        lines.append(f"recommended escalation threshold: margin <= {rec['threshold']:g} catches {rec['caught']}/"
                     f"{rec['errors']} judge errors (target {target:g}) by escalating {rec['escalated']} verdicts "
                     f"({rec['escalate_fraction']:.1%})")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="judge_calibration")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report", help="calibrate a judge against gold and recommend an escalation threshold")
    rp.add_argument("--judge", required=True, help="judge verdict JSONL")
    rp.add_argument("--gold", required=True, help="gold JSONL")
    rp.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="grade >= this counts as correct (1..3)")
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    rp.add_argument("--target-capture", type=float, default=0.8)
    rp.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        judge_rows, gold_rows = _read_jsonl(args.judge), _read_jsonl(args.gold)
        cal = calibrate(judge_rows, gold_rows, threshold=args.threshold, seed=args.seed, n_boot=args.boot)
        gold = normalize_gold(gold_rows, args.threshold)
        rec = recommend_threshold(normalize_judge(judge_rows)[0], gold, args.target_capture)
    except (CalibrationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"calibration": cal, "recommended_threshold": rec}, sort_keys=True) if args.json
          else render_report(cal, rec, args.target_capture))
    return 0


if __name__ == "__main__":
    sys.exit(main())
