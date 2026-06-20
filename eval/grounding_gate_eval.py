"""
Longitudinal eval for the deep-think-loci grounding gate.

Measures whether the cosine grounding gate still separates on-topic evidence
from cross-target RAG-bleed on the labeled corpus (deep_think_loci/grounding/
grounding_dataset.jsonl, topical pairs). Reuses the harness's embed / upsert /
collection helpers so scores land in the same `eval_scores` Qdrant collection
and trend alongside the grounding-pipeline scores.

- Live: re-embeds each unique finding text (catches embedder/threshold drift),
  computes pair cosines, persists recall / bleed_rejection / f1 / accuracy / auc.
- Dry run (HARNESS_DRY_RUN=1): uses the cosines stored in the dataset — no
  Qdrant/Ollama, CI-safe — and prints the metrics without persisting.

Run via eval/run_eval.sh (which invokes both harness.py and this), or directly
with HERMES_PY. Threshold from $DTL_GROUND_THRESHOLD (default 0.59).
"""
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import harness  # reuse embed / ensure_collection / upsert_score / DRY_RUN / config

THRESHOLD = float(os.environ.get("DTL_GROUND_THRESHOLD", "0.59"))
DATASET = Path(os.environ.get(
    "DTL_GROUNDING_DATASET",
    str(Path(__file__).resolve().parent.parent / "deep_think_loci/grounding/grounding_dataset.jsonl"),
))
CATEGORY = "deep_think_loci"


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cosines_live(pairs):
    """Re-embed unique texts once, return cosine per pair (catches drift)."""
    texts = sorted({t for p in pairs for t in (p["claim"], p["evidence"])})
    vec = {t: _unit(harness.embed(t)) for t in texts}
    return [sum(a * b for a, b in zip(vec[p["claim"]], vec[p["evidence"]])) for p in pairs]


def _metrics(labels, cos, thr):
    keep = [c >= thr for c in cos]
    pos = [l for l in labels if l == 1]
    neg = [l for l in labels if l == 0]
    tp = sum(1 for l, k in zip(labels, keep) if l == 1 and k)
    fp = sum(1 for l, k in zip(labels, keep) if l == 0 and k)
    recall = tp / len(pos) if pos else 0.0
    bleed_rejection = sum(1 for l, k in zip(labels, keep) if l == 0 and not k) / len(neg) if neg else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = sum(1 for l, k in zip(labels, keep) if (l == 1) == k) / len(labels) if labels else 0.0
    # rank-based AUC of cosine separating on-topic from bleed
    ps = [c for l, c in zip(labels, cos) if l == 1]
    ns = [c for l, c in zip(labels, cos) if l == 0]
    auc = (sum((p > n) + 0.5 * (p == n) for p in ps for n in ns) / (len(ps) * len(ns))) if ps and ns else float("nan")
    return {"recall": recall, "bleed_rejection": bleed_rejection, "f1": f1, "accuracy": accuracy, "auc": auc}


def run():
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not DATASET.exists():
        print(f"[gate-eval] dataset not found: {DATASET}")
        return
    pairs = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
    topical = [p for p in pairs if p.get("signal") == "topical"]
    if not topical:
        print("[gate-eval] no topical pairs in dataset")
        return
    labels = [int(p["label"]) for p in topical]

    print(f"[gate-eval] run_date={run_date} pairs={len(topical)} thr={THRESHOLD} dry_run={harness.DRY_RUN}")
    if harness.DRY_RUN:
        cos = [float(p.get("cos", 0.0)) for p in topical]  # stored cosines, no network
    else:
        harness.ensure_collection()
        cos = _cosines_live(topical)

    m = _metrics(labels, cos, THRESHOLD)
    for name, val in m.items():
        print(f"  {name:16} {val:.3f}")

    if not harness.DRY_RUN:
        vec = harness.embed("deep_think_loci grounding gate eval")
        for name, val in m.items():
            if val == val:  # skip NaN
                harness.upsert_score(
                    task_id=f"dtl.grounding_gate.{name}",
                    task_name=f"grounding gate {name} @cos>={THRESHOLD}",
                    category=CATEGORY,
                    score=float(val),
                    run_date=run_date,
                    context_preview=f"{len(topical)} topical pairs from {DATASET.name}",
                    vector=vec,
                )
        print(f"[gate-eval] persisted {sum(1 for v in m.values() if v == v)} scores to {harness.EVAL_COLLECTION}")


if __name__ == "__main__":
    run()
