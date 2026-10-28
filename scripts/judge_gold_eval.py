#!/usr/bin/env python3
"""Judge-vs-gold report and gold-CANDIDATE builder (Tripwire-Governor S12). Makes no model call.

  judge_gold_eval.py gate --gold gold.jsonl [--json]
      Read a gold-set JSONL and print the judge gate report (false-accept, kappa, counts, per-bucket false-accept).
      Exit 0 when the gate PASSES, 1 when it fails (the usual case today: the judge stays non-gating), 2 on bad input.
      A row: {"id", "gold_correct": bool, "judge_accept": bool|null, "hard_negative": bool, "bucket", "answerer",
      "judge_model"}. ``judge_accept`` is the judge's RECORDED verdict; a live judge is never called from here (the
      Python API ``evaluate_file(path, judge=callable)`` takes an injected one for callers that own a model budget).

  judge_gold_eval.py candidates --outcomes outcomes.jsonl --out candidates.jsonl --disagreements dis.jsonl
      Decision 4: seed the gold set from deterministic validator verdicts and existing graded outcomes WITHOUT
      labelling anything. Each outcome row ({"id", "text", "finish_reason", "tool_calls", "stream", "bucket",
      "judge_accept", "existing_grade", "grader", "answerer"}) is run through the validators. The candidate file
      carries the validator verdict, its reasons, the judge's recorded verdict and any pre-existing grade copied
      verbatim, and ``"gold_correct": null`` on every row: only the owner fills that in. The disagreements file lists
      the rows where the judge and the validators disagree (with a 300-char excerpt), for the owner to settle.
      Existing output files are never overwritten.

Observe-only: nothing here is read by a live path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, TextIO, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp"))
import quality_validators as qv  # noqa: E402

EXCERPT_CHARS = 300
KIND_ACCEPT_FAIL = "judge_accept_validator_fail"       # strong: the validator proves the answer is bad
KIND_REJECT_PASS = "judge_reject_validator_pass"       # weak: a structurally fine answer the judge rejected


def read_jsonl(path: Path) -> Tuple[List[dict], int]:
    """Rows that are JSON objects, and how many lines were not (blank lines are ignored)."""
    rows: List[dict] = []
    bad = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            bad += 1
    return rows, bad


def evaluate_file(path: Path, judge: Optional[Callable[[dict], Optional[bool]]] = None) -> qv.GateReport:
    """Gate report for a gold JSONL. ``judge(row) -> bool|None`` replaces the recorded ``judge_accept`` when given;
    a judge that raises counts as no verdict for that row (reported as unusable), never as an accept."""
    rows, bad = read_jsonl(path)
    items: List[qv.GoldItem] = []
    malformed = bad
    for row in rows:
        if judge is not None:
            try:
                verdict = judge(row)
            except Exception:
                verdict = None
            row = {**row, "judge_accept": verdict if isinstance(verdict, bool) else None}
        try:
            items.append(qv.GoldItem.from_mapping(row))
        except qv.GoldFormatError:
            malformed += 1
    return qv.evaluate_gate(items, malformed=malformed)


def _bool_or_none(v: object) -> Optional[bool]:
    return v if isinstance(v, bool) else None


def build_candidates(rows: Iterable[dict]) -> Tuple[List[dict], List[dict]]:
    """(candidate rows, disagreement rows). No candidate row carries a label: ``gold_correct`` is always null."""
    candidates: List[dict] = []
    disagreements: List[dict] = []
    for row in rows:
        result = qv.validate(qv.CallOutcome.from_mapping(row))
        judge = _bool_or_none(row.get("judge_accept"))
        cand = {"id": str(row.get("id", "")), "bucket": str(row.get("bucket") or ""),
                "answerer": str(row.get("answerer") or ""), "validator_verdict": result.verdict,
                "validator_reasons": list(result.reasons), "judge_accept": judge,
                "existing_grade": _bool_or_none(row.get("existing_grade")), "grader": str(row.get("grader") or ""),
                "candidate_hard_negative": result.verdict == qv.FAIL, "gold_correct": None}
        candidates.append(cand)
        kind = ""
        if judge is True and result.verdict == qv.FAIL:
            kind = KIND_ACCEPT_FAIL
        elif judge is False and result.verdict == qv.PASS:
            kind = KIND_REJECT_PASS
        if kind:
            text = row.get("text")
            disagreements.append({"id": cand["id"], "kind": kind, "validator_reasons": cand["validator_reasons"],
                                  "judge_accept": judge, "bucket": cand["bucket"],
                                  "excerpt": (text if isinstance(text, str) else "")[:EXCERPT_CHARS]})
    return candidates, disagreements


def _write_new(path: Path, rows: Sequence[dict]) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as fh:      # "x": never overwrite
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def format_report(rep: qv.GateReport) -> str:
    def pct(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x:.3f}"
    lines = [f"judge gate: {'PASS' if rep.passed else 'FAIL'}",
             f"  items usable={rep.n} unusable={rep.n_unusable} gold-negative={rep.n_gold_negative} "
             f"hard-negative={rep.n_hard_negative}",
             f"  confusion tp={rep.tp} fp={rep.fp} fn={rep.fn} tn={rep.tn}",
             f"  false_accept={pct(rep.false_accept)} (max {float(qv.FALSE_ACCEPT_MAX):.2f}) "
             f"false_accept_hard={pct(rep.false_accept_hard)} kappa={pct(rep.kappa)} (min {float(qv.KAPPA_MIN):.1f}) "
             f"agreement={pct(rep.agreement)}"]
    for bucket, fa in rep.by_bucket.items():
        lines.append(f"  bucket {bucket}: false_accept={pct(fa)}")
    if rep.reasons:
        lines.append("  failed because: " + ", ".join(rep.reasons))
        lines.append("  the judge gates nothing until this passes")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None, out: Optional[TextIO] = None) -> int:
    out = out or sys.stdout
    ap = argparse.ArgumentParser(prog="judge_gold_eval", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="print the judge gate report for a gold JSONL")
    g.add_argument("--gold", required=True, type=Path)
    g.add_argument("--json", action="store_true")
    c = sub.add_parser("candidates", help="build a gold CANDIDATE file and a judge-vs-validator disagreement list")
    c.add_argument("--outcomes", required=True, type=Path)
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--disagreements", required=True, type=Path)
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0) if exc.code in (0, None) else 2
    try:
        if args.cmd == "gate":
            rep = evaluate_file(args.gold)
            print(json.dumps(rep.to_dict(), sort_keys=True) if args.json else format_report(rep), file=out)
            return 0 if rep.passed else 1
        for p in (args.out, args.disagreements):
            if p.exists():
                print(f"refusing to overwrite {p}", file=out)
                return 2
        rows, bad = read_jsonl(args.outcomes)
        cands, dis = build_candidates(rows)
        _write_new(args.out, cands)
        _write_new(args.disagreements, dis)
        print(f"candidates={len(cands)} disagreements={len(dis)} unreadable_lines={bad} labelled=0", file=out)
        return 0
    except OSError as exc:
        print(f"error: {exc}", file=out)
        return 2


if __name__ == "__main__":
    sys.exit(main())
