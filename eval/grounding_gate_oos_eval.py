"""
Out-of-sample (leave-one-run-out) validation of the grounding gate.

The in-sample query->finding eval (grounding_gate_qf_eval.py) trains and tests on
the same runs' findings, so its numbers are optimistic. This is the honest check:
for each held-out run R, train the bleed-detector on the OTHER runs' findings,
then evaluate cosine vs that model on R (which the model never saw). Reports
per-fold and mean model-vs-cosine F1/accuracy — whether the model GENERALIZES.

On-demand, live-only (needs embeddings + OLLAMA_URL). Print-only (no persist).
Run: OLLAMA_URL=... HERMES_PY ... eval/grounding_gate_oos_eval.py
"""
import glob
import itertools
import json
import os
import statistics
from collections import defaultdict

import harness
from grounding_gate_eval import _metrics
from grounding_gate_qf_eval import _focus_map, _target, _unit

THRESHOLD = float(os.environ.get("DTL_GROUND_THRESHOLD", "0.59"))
CORPUS = os.environ.get("DTL_CORPUS_GLOB", os.path.expanduser("~/.hermes/memory-sessions/dt-loci-*/findings.jsonl"))


def run():
    if not harness.OLLAMA_URL:
        print("[oos] OLLAMA_URL unset — needs embeddings; skipping.")
        return
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    focus = _focus_map()
    runs = defaultdict(list)
    for f in glob.glob(CORPUS):
        run_id = f.split("/")[-2]
        for line in open(f, errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = _target(r)
            if t:
                runs[run_id].append({"text": (r.get("text") or "")[:2000], "target": t})
    run_ids = sorted(runs)
    if len(run_ids) < 2:
        print(f"[oos] need >=2 runs, found {len(run_ids)}")
        return

    # embed every unique finding text + each target focus once
    texts = sorted({x["text"] for fs in runs.values() for x in fs})
    fv = {tx: np.array(_unit(harness.embed(tx))) for tx in texts}
    targets = sorted({x["target"] for fs in runs.values() for x in fs})
    qv = {t: np.array(_unit(harness.embed(focus.get(t, t)))) for t in targets}

    def feat(a, b):
        return np.concatenate([np.abs(a - b), a * b, [float(a @ b)]])

    folds = []
    for held in run_ids:
        train = [x for rid in run_ids if rid != held for x in runs[rid]]
        test = runs[held]
        # train pair-model (same-target=1 / cross-target=0) on the OTHER runs
        tt = [(x["text"], x["target"]) for x in train]
        X, Y = [], []
        for (ta, ga), (tb, gb) in itertools.combinations(tt, 2):
            X.append(feat(fv[ta], fv[tb]))
            Y.append(1 if ga == gb else 0)
        if len(set(Y)) < 2:
            continue
        clf = LogisticRegression(max_iter=2000).fit(np.array(X), np.array(Y))
        # query->finding eval on the held-out run (unseen)
        labels, cosv, feats = [], [], []
        for t in targets:
            for x in test:
                cv = fv[x["text"]]
                cos = float(qv[t] @ cv)
                labels.append(1 if x["target"] == t else 0)
                cosv.append(cos)
                feats.append(feat(qv[t], cv))
        if not labels or sum(labels) == 0:
            continue
        cm = _metrics(labels, cosv, THRESHOLD)
        proba = clf.predict_proba(np.array(feats))[:, 1].tolist()
        mm = _metrics(labels, proba, 0.5)
        folds.append((held, cm, mm))
        print(f"  held={held} (n={len(test)}): cosine f1={cm['f1']:.3f} acc={cm['accuracy']:.3f} "
              f"| model f1={mm['f1']:.3f} acc={mm['accuracy']:.3f}")

    if not folds:
        print("[oos] no usable folds")
        return
    cf = statistics.mean(c["f1"] for _, c, _ in folds)
    ca = statistics.mean(c["accuracy"] for _, c, _ in folds)
    mf = statistics.mean(m["f1"] for _, _, m in folds)
    ma = statistics.mean(m["accuracy"] for _, _, m in folds)
    print(f"[oos] MEAN over {len(folds)} folds: cosine f1={cf:.3f} acc={ca:.3f} | model f1={mf:.3f} acc={ma:.3f}")
    generalizes = mf > cf and ma >= ca
    print(f"[oos] VERDICT (out-of-sample): trained model "
          f"{'GENERALIZES — beats cosine' if generalizes else 'does NOT clearly beat cosine'} "
          f"(keep model default)" if generalizes else "(consider reverting gate to cosine default: --no-model)")


if __name__ == "__main__":
    run()
