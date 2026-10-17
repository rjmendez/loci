#!/usr/bin/env python3
"""Score the grounding model on a split that does not leak.

train.py reports 10-fold CV over PAIRS. A finding therefore appears in train and
test on every fold, and the number is optimistic: measured on this corpus,
GradientBoosting scores 0.944 pair-level and 0.908 when findings are split first,
against a cosine baseline that barely moves (0.864 -> 0.878). Two thirds of the
apparent margin over cosine is the leak.

So this splits FINDINGS, builds pairs inside each side, and reports the honest
margin. It is the measurement to use when deciding whether more corpus helped,
because a pair-level number rises with corpus size whether or not the model
learned anything.

Paired across seeds: the same seed gives every arm the same split, so
between-seed variance cancels and a small real effect is still visible.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "deep_think_loci" / "grounding"))
sys.path.insert(0, str(HERE))
import features as F           # noqa: E402
import train as T              # noqa: E402


def _topic(rec: dict) -> str:
    tags = rec.get("tags") or []
    tg = {t.split(":", 1)[0]: t.split(":", 1)[1] for t in tags if ":" in t}
    if tg.get("dt_target"):
        return tg["dt_target"]
    if tg.get("dt_phase") in ("final", "adversarial"):
        return "synthesis:" + tg["dt_phase"]
    return tg.get("dt_phase", "unknown")


def load_findings(pattern: str) -> list[dict]:
    """One row per id, first with usable text. Same rule as the builder — a
    corpus that disagrees with the builder measures the wrong thing."""
    by_id: dict[str, dict] = {}
    rows = 0
    for f in sorted(set(glob.glob(os.path.expanduser(pattern)))):
        for line in open(f, errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows += 1
            fid, text = r.get("id"), (r.get("text") or "").strip()[:2000]
            if not fid or not text or fid in by_id:
                continue
            by_id[fid] = {"text": text, "topic": _topic(r)}
    print(f"[holdout] {len(by_id)} findings from {rows} rows", flush=True)
    return list(by_id.values())


def pairs(recs, idxs, emb, seed, neg_ratio=2):
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    for a, b in itertools.combinations(idxs, 2):
        (pos if recs[a]["topic"] == recs[b]["topic"] else neg).append((a, b))
    rng.shuffle(neg)
    neg = neg[: len(pos) * neg_ratio]
    if not pos:
        return None
    ia = np.array([a for a, _ in pos + neg])
    ib = np.array([b for _, b in pos + neg])
    ta = [recs[i]["text"] for i in ia]
    tb = [recs[i]["text"] for i in ib]
    X = F.make_features(ta, tb, emb[ia], emb[ib])
    y = np.array([1] * len(pos) + [0] * len(neg))
    cos = (emb[ia] * emb[ib]).sum(1)
    return X, y, cos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--findings", default=os.path.expanduser(
        "~/.hermes/memory-sessions/dt-loci-*/findings.jsonl"))
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_BASE_URL", ""))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--model", default="both", choices=("lr", "gbc", "both"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    recs = load_findings(a.findings)
    if len(recs) < 30:
        print(f"[holdout] only {len(recs)} findings — too few to split honestly", file=sys.stderr)
        return 1

    cache = T._load_cache()
    emb = T.embed_texts([r["text"] for r in recs], a.ollama, cache)
    T._save_cache(cache)

    arms = {"LogisticRegression": lambda: LogisticRegression(C=0.1, max_iter=2000,
                                                             class_weight="balanced")}
    if a.model in ("gbc", "both"):
        arms["GradientBoostingClassifier"] = lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
    if a.model == "gbc":
        arms.pop("LogisticRegression")

    res: dict[str, list] = {k: [] for k in arms}
    base = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(recs))
        n_test = int(len(recs) * a.test_frac)
        te, tr = list(order[:n_test]), list(order[n_test:])
        pte, ptr = pairs(recs, te, emb, seed), pairs(recs, tr, emb, seed)
        if pte is None or ptr is None:
            print(f"[holdout] seed {seed}: a side had no positive pairs — skipped")
            continue
        Xte, yte, coste = pte
        Xtr, ytr, _ = ptr
        base.append(max(f1_score(yte, (coste > t).astype(int), zero_division=0)
                        for t in np.linspace(0.2, 0.9, 71)))
        for name, mk in arms.items():
            clf = mk().fit(Xtr, ytr)
            res[name].append(f1_score(yte, clf.predict(Xte), zero_division=0))
        print(f"[holdout] seed {seed} done", flush=True)

    out = {"findings": len(recs), "seeds": len(base),
           "cosine_f1_mean": float(np.mean(base)), "cosine_f1_std": float(np.std(base)),
           "models": {}}
    print(f"\n  {'model':<28}{'F1':>8}{'sd':>8}{'over cosine':>14}")
    print(f"  {'cosine (oracle thr)':<28}{np.mean(base):>8.3f}{np.std(base):>8.3f}")
    for name, v in res.items():
        if not v:
            continue
        delta = np.array(v) - np.array(base)          # paired: seed variance cancels
        out["models"][name] = {"f1_mean": float(np.mean(v)), "f1_std": float(np.std(v)),
                               "over_cosine_mean": float(delta.mean()),
                               "over_cosine_std": float(delta.std())}
        print(f"  {name:<28}{np.mean(v):>8.3f}{np.std(v):>8.3f}"
              f"{delta.mean():>+10.3f} ±{delta.std():.3f}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"\n[holdout] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
