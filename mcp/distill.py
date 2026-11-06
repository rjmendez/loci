"""Distillation tooling: teacher labels -> splits -> D10 trainer input -> promote/hold gate -> label budget.

Pure and offline: no model call, no GPU, no write outside the export dir. Promotion is a manual step;
``decide`` only returns a decision with reasons. Criteria are data (a JSON file), never code constants.

Row formats (JSONL):
  gold / teacher:  {"query_id"|"id", "claim", "evidence", "label"|"gold_correct" (0/1/bool), ["teacher_label"]}
  Gold wins over a teacher label for the same query id. Splits are by query id, so one query never
  spans two splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SPLITS = ("train", "validation", "test")
CRITERIA_KEYS = ("metric", "threshold", "min_margin_over_incumbent", "max_teacher_gap_on_validation",
                 "min_test_rows", "min_validation_rows")
METRICS = ("accuracy", "f1", "auc", "precision", "recall")


class DistillError(ValueError):
    """Malformed input rows, criteria or curve."""


# ------------------------------------------------------------------ rows

def _to_label(v: Any, where: str) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int) and v in (0, 1):
        return v
    raise DistillError(f"{where}: label must be 0, 1 or bool, got {v!r}")


def normalize_rows(raw: Iterable[Mapping[str, Any]], source: str) -> list[dict[str, Any]]:
    """Validate rows and assign ``row_id`` = ``<query_id>#<k>`` (k = order within the query)."""
    out: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for n, r in enumerate(raw, start=1):
        where = f"{source} row {n}"
        qid = r.get("query_id", r.get("id"))
        if qid is None or str(qid) == "":
            raise DistillError(f"{where}: query_id (or id) required")
        qid = str(qid)
        claim, ev = r.get("claim"), r.get("evidence")
        if not isinstance(claim, str) or not isinstance(ev, str) or not claim or not ev:
            raise DistillError(f"{where}: claim and evidence must be non-empty strings")
        lab = r["label"] if "label" in r else r.get("gold_correct")
        if lab is None:
            raise DistillError(f"{where}: label (or gold_correct) required")
        row = {"query_id": qid, "row_id": f"{qid}#{seen.get(qid, 0)}", "claim": claim, "evidence": ev,
               "label": _to_label(lab, where), "source": source}
        seen[qid] = seen.get(qid, 0) + 1
        if r.get("teacher_label") is not None:
            row["teacher_label"] = _to_label(r["teacher_label"], where)
        out.append(row)
    return out


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = Path(path).read_bytes()
    rows = []
    for n, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DistillError(f"{path} line {n}: invalid JSON ({e.msg})") from e
    return rows, hashlib.sha256(raw).hexdigest()


def merge_sources(gold: Sequence[dict], teacher: Sequence[dict]) -> tuple[list[dict], int]:
    """Gold rows plus teacher rows whose query id has no gold row. Returns (rows, teacher_rows_dropped)."""
    gold_ids = {r["query_id"] for r in gold}
    kept = [r for r in teacher if r["query_id"] not in gold_ids]
    return list(gold) + kept, len(teacher) - len(kept)


# ------------------------------------------------------------------ splits

def split_rows(rows: Sequence[dict], seed: int, validation_frac: float = 0.15,
               test_frac: float = 0.15) -> dict[str, list[dict]]:
    """Split by query id; membership depends only on (seed, id set), not on input order."""
    if not (0 < validation_frac < 1 and 0 < test_frac < 1 and validation_frac + test_frac < 1):
        raise DistillError("validation_frac and test_frac must be in (0,1) and sum below 1")
    ids = sorted({r["query_id"] for r in rows})
    if len(ids) < 3:
        raise DistillError(f"need at least 3 distinct query ids to fill three splits, got {len(ids)}")
    ids.sort(key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    n_val = max(1, round(len(ids) * validation_frac))
    n_test = max(1, round(len(ids) * test_frac))
    which = {}
    for k, i in enumerate(ids):
        which[i] = "test" if k < n_test else "validation" if k < n_test + n_val else "train"
    out: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for r in rows:
        out[which[r["query_id"]]].append(r)
    for s in SPLITS:
        out[s].sort(key=lambda r: (r["query_id"], r["row_id"]))
    if not out["train"]:
        raise DistillError("train split is empty; lower validation_frac/test_frac")
    return out


def _jsonl(rows: Sequence[Mapping]) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)


def d10_rows(rows: Sequence[dict]) -> list[dict]:
    """Exactly the D10 trainer's dataset rows: claim, evidence, label (0/1), signal."""
    return [{"claim": r["claim"], "evidence": r["evidence"], "label": r["label"], "signal": r["source"]}
            for r in rows]


def build_manifest(splits: Mapping[str, Sequence[dict]], seed: int, sources: Mapping[str, Any],
                   teacher_dropped: int, fracs: Mapping[str, float]) -> dict:
    m: dict[str, Any] = {"schema": "loci-distill-export/v1", "seed": seed, "fractions": dict(fracs),
                         "source": dict(sources), "teacher_rows_dropped_gold_wins": teacher_dropped,
                         "splits": {}}
    for s in SPLITS:
        rows = splits[s]
        m["splits"][s] = {
            "rows": len(rows), "queries": len({r["query_id"] for r in rows}),
            "labels": {"0": sum(r["label"] == 0 for r in rows), "1": sum(r["label"] == 1 for r in rows)},
            "by_source": {k: sum(r["source"] == k for r in rows) for k in sorted({r["source"] for r in rows})},
            "sha256": hashlib.sha256(_jsonl(rows).encode()).hexdigest()}
    return m


def export(gold_path: Path, out_dir: Path, seed: int, teacher_path: Path | None = None,
           validation_frac: float = 0.15, test_frac: float = 0.15) -> dict:
    """Write train/validation/test.jsonl, d10_train.jsonl and manifest.json under ``out_dir``."""
    graw, ghash = read_jsonl(gold_path)
    sources: dict[str, Any] = {"gold": {"path": str(gold_path), "sha256": ghash, "rows": len(graw)}}
    teacher: list[dict] = []
    if teacher_path is not None:
        traw, thash = read_jsonl(teacher_path)
        teacher = normalize_rows(traw, "teacher")
        sources["teacher"] = {"path": str(teacher_path), "sha256": thash, "rows": len(traw)}
    rows, dropped = merge_sources(normalize_rows(graw, "gold"), teacher)
    splits = split_rows(rows, seed, validation_frac, test_frac)
    manifest = build_manifest(splits, seed, sources, dropped,
                              {"validation": validation_frac, "test": test_frac})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        (out_dir / f"{s}.jsonl").write_bytes(_jsonl(splits[s]).encode("utf-8"))
    (out_dir / "d10_train.jsonl").write_bytes(_jsonl(d10_rows(splits["train"])).encode("utf-8"))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def trainer_command(dataset: Path, loci_commit: str, out: Path, embed_url: str) -> str:
    """Command line for the D10 trainer on the exported train rows. Not executed here."""
    return " ".join(shlex.quote(str(p)) for p in (
        "python", "deep_think_loci/grounding/train_d10_pair_gate.py", "--loci-commit", loci_commit,
        "--dataset", Path(dataset).as_posix(), "--out", Path(out).as_posix(), "--embed-url", embed_url))


# ------------------------------------------------------------------ student gate

def _auc(scores: Sequence[float], ys: Sequence[int]) -> float | None:
    pos = [s for s, y in zip(scores, ys) if y]
    neg = [s for s, y in zip(scores, ys) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def evaluate(scores: Mapping[str, float], gold_split: Sequence[dict], threshold: float) -> dict:
    """Metrics of ``scores`` (row_id -> score; predict 1 when score >= threshold) on a gold split."""
    if not gold_split:
        raise DistillError("gold split is empty")
    missing = [r["row_id"] for r in gold_split if r["row_id"] not in scores]
    if missing:
        raise DistillError(f"{len(missing)} rows have no score, first: {missing[0]}")
    ys = [r["label"] for r in gold_split]
    ss = [float(scores[r["row_id"]]) for r in gold_split]
    pred = [int(s >= threshold) for s in ss]
    tp = sum(1 for p, y in zip(pred, ys) if p and y)
    fp = sum(1 for p, y in zip(pred, ys) if p and not y)
    fn = sum(1 for p, y in zip(pred, ys) if not p and y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(ys), "accuracy": sum(p == y for p, y in zip(pred, ys)) / len(ys),
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "auc": _auc(ss, ys)}


def load_criteria(path: Path) -> dict:
    c = _load_json(path)
    if not isinstance(c, dict):
        raise DistillError(f"criteria {path}: must be a JSON object")
    missing = [k for k in CRITERIA_KEYS if k not in c]
    if missing:
        raise DistillError(f"criteria {path}: missing keys {missing}")
    if c["metric"] not in METRICS:
        raise DistillError(f"criteria {path}: unknown metric {c['metric']!r}")
    for k in CRITERIA_KEYS[1:]:
        v = c[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise DistillError(f"criteria {path}: {k} must be a finite number, got {v!r}")
    return c


def decide(criteria: Mapping[str, Any], test_split: Sequence[dict], val_split: Sequence[dict],
           student_test: Mapping[str, float], incumbent_test: Mapping[str, float],
           student_val: Mapping[str, float]) -> dict:
    """Promote only when every criterion holds. Returns {decision, reasons, metrics}; writes nothing.

    1. student metric on test >= incumbent metric on test + min_margin_over_incumbent;
    2. student metric on validation >= teacher's metric vs gold on validation - max_teacher_gap_on_validation
       (the teacher's label is each validation row's ``teacher_label``);
    3. split sizes at least min_test_rows / min_validation_rows.
    """
    metric, thr = criteria["metric"], criteria["threshold"]
    reasons: list[str] = []
    if len(test_split) < criteria["min_test_rows"]:
        reasons.append(f"test split has {len(test_split)} rows, need {criteria['min_test_rows']}")
    if len(val_split) < criteria["min_validation_rows"]:
        reasons.append(f"validation split has {len(val_split)} rows, need {criteria['min_validation_rows']}")
    no_teacher = [r["row_id"] for r in val_split if "teacher_label" not in r]
    if no_teacher:
        reasons.append(f"{len(no_teacher)} validation rows lack teacher_label "
                       "(gold validation rows must carry it too)")
    st = evaluate(student_test, test_split, thr)
    inc = evaluate(incumbent_test, test_split, thr)
    sv = evaluate(student_val, val_split, thr)
    metrics: dict[str, Any] = {"student_test": st, "incumbent_test": inc, "student_validation": sv}
    for name, m in (("student_test", st), ("incumbent_test", inc), ("student_validation", sv)):
        if m[metric] is None:
            reasons.append(f"{name} {metric} undefined (single-class split)")
    if not no_teacher:
        tv = evaluate({r["row_id"]: float(r["teacher_label"]) for r in val_split}, val_split, 0.5)
        metrics["teacher_validation"] = tv
        if tv[metric] is None:
            reasons.append(f"teacher_validation {metric} undefined (single-class split)")
        elif sv[metric] is not None and sv[metric] < tv[metric] - criteria["max_teacher_gap_on_validation"]:
            reasons.append(f"validation {metric} {sv[metric]:.4f} is below teacher {tv[metric]:.4f} "
                           f"by more than {criteria['max_teacher_gap_on_validation']}")
    if st[metric] is not None and inc[metric] is not None and \
            st[metric] < inc[metric] + criteria["min_margin_over_incumbent"]:
        reasons.append(f"test {metric} {st[metric]:.4f} does not beat incumbent {inc[metric]:.4f} "
                       f"by {criteria['min_margin_over_incumbent']}")
    return {"decision": "hold" if reasons else "promote",
            "reasons": reasons or ["all criteria met"], "metrics": metrics}


# ------------------------------------------------------------------ label budget

def plan_budget(cap_usd: float, cost_per_label_usd: float, curve: Sequence[Sequence[float]],
                min_gain_per_step: float, step_labels: int) -> dict:
    """Affordable labels and the learning-curve stop point.

    ``curve`` = [(n_labels, metric)] so far. Slope is gain per ``step_labels`` between consecutive points;
    the curve is saturated when the last slope is below ``min_gain_per_step``. Otherwise the slope is
    assumed to decay geometrically at its last observed ratio (needs 3 points) and the stop is where it
    crosses the floor. ``stop_labels`` is None when it cannot be estimated.
    """
    if cap_usd < 0 or cost_per_label_usd <= 0 or step_labels <= 0 or min_gain_per_step <= 0:
        raise DistillError("cap_usd >= 0 and cost_per_label_usd, step_labels, min_gain_per_step > 0 required")
    if not all(map(math.isfinite, (cap_usd, cost_per_label_usd, min_gain_per_step))):
        raise DistillError("cap_usd, cost_per_label_usd, min_gain_per_step must be finite")
    pts = sorted((int(n), float(m)) for n, m in curve)
    if not all(math.isfinite(m) for _, m in pts):
        raise DistillError("curve has a non-finite metric")
    if len({n for n, _ in pts}) != len(pts):
        raise DistillError("curve has duplicate label counts")
    have = pts[-1][0] if pts else 0
    affordable = math.floor(cap_usd / cost_per_label_usd)
    out: dict[str, Any] = {"affordable_labels": affordable, "labels_so_far": have, "saturated": False,
                           "stop_labels": None, "note": ""}
    if len(pts) < 2:
        out["note"] = "need at least 2 curve points to judge saturation"
        return out
    slopes = [(m2 - m1) / (n2 - n1) * step_labels for (n1, m1), (n2, m2) in zip(pts, pts[1:])]
    last = slopes[-1]
    out["last_slope_per_step"] = last
    if last < min_gain_per_step:
        out.update(saturated=True, stop_labels=have, note="last slope below floor; more labels will not pay")
    elif len(slopes) >= 2 and 0 < last < slopes[-2]:
        r = last / slopes[-2]
        k = math.ceil(math.log(min_gain_per_step / last) / math.log(r))
        out.update(stop_labels=have + k * step_labels, note=f"extrapolated, slope decay ratio {r:.3f}")
    else:
        out["note"] = "slope is not decaying; no stop point estimable"
    if out["stop_labels"] is not None:
        out["additional_labels_needed"] = max(0, out["stop_labels"] - have)
        out["stop_within_budget"] = out["additional_labels_needed"] <= affordable
    return out


# ------------------------------------------------------------------ CLI

def _load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise DistillError(f"{path}: cannot read ({e.strerror or e})") from e
    except ValueError as e:
        raise DistillError(f"{path}: invalid JSON ({e})") from e


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="distill", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write splits, D10 train rows and manifest; print the trainer command")
    e.add_argument("--gold", required=True, type=Path)
    e.add_argument("--teacher", type=Path)
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--seed", required=True, type=int)
    e.add_argument("--validation-frac", type=float, default=0.15)
    e.add_argument("--test-frac", type=float, default=0.15)
    e.add_argument("--loci-commit", required=True)
    e.add_argument("--embed-url", required=True)
    e.add_argument("--model-out", required=True, type=Path, help="trainer output dir (a candidate, not the pinned artifact)")
    g = sub.add_parser("gate", help="promote/hold decision from score files {row_id: score}; exit 1 on hold")
    for name in ("criteria", "test", "validation", "student-test", "incumbent-test", "student-validation"):
        g.add_argument(f"--{name}", required=True, type=Path)
    p = sub.add_parser("plan", help="label budget: affordable labels and learning-curve stop point")
    p.add_argument("--cap-usd", required=True, type=float)
    p.add_argument("--cost-per-label-usd", required=True, type=float)
    p.add_argument("--curve", required=True, type=Path, help="JSON [[n_labels, metric], ...]")
    p.add_argument("--min-gain-per-step", required=True, type=float)
    p.add_argument("--step-labels", required=True, type=int)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "export":
            m = export(a.gold, a.out, a.seed, a.teacher, a.validation_frac, a.test_frac)
            print(json.dumps(m, indent=2, sort_keys=True))
            print(trainer_command(a.out / "d10_train.jsonl", a.loci_commit, a.model_out, a.embed_url))
            return 0
        if a.cmd == "gate":
            res = decide(load_criteria(a.criteria), read_jsonl(a.test)[0], read_jsonl(a.validation)[0],
                         _load_json(a.student_test), _load_json(a.incumbent_test),
                         _load_json(a.student_validation))
            print(json.dumps(res, indent=2, sort_keys=True))
            return 0 if res["decision"] == "promote" else 1
        res = plan_budget(a.cap_usd, a.cost_per_label_usd, _load_json(a.curve),
                          a.min_gain_per_step, a.step_labels)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    except DistillError as ex:
        print(f"distill: {ex}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
