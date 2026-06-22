"""
Faithful query->finding eval for the deep-think-loci grounding gate.

The deployed gate operates query->finding (a target's focus query vs each
candidate finding), NOT finding<->finding (which is the classifier's training
task, measured by grounding_gate_eval.py). This measures the gate's ACTUAL
operation on the corpus and compares the cosine threshold against the trained
classifier.

IN-SAMPLE DIAGNOSTIC ONLY: the trained model is fit on this same corpus, so its
metrics here are optimistic (AUC can read ~1.0 purely from memorization). This
does NOT decide the gate default — that is governed by the out-of-sample
grounding_gate_oos_eval.py (leave-one-run-out), which is the eval that catches the
in-sample mirage and reverts the swap when the model fails to generalize.

For every (target focus query, finding) pair: cosine = sim(embed(focus),
embed(finding)); label = 1 if the finding belongs to that target else 0 (bleed).
Reports recall / bleed_rejection / f1 / accuracy / auc for cosine and (if present)
the trained model, each at a fixed AND best-F1 threshold (so the F1 comparison is
fair), plus an in-sample-only note that defers the swap decision to the OOS eval.

Live-only (needs embeddings). Requires OLLAMA_URL. Persists scores
dtl.gate_qf.{cosine,model}.{metric} unless HARNESS_DRY_RUN=1.
"""
import glob
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import harness
from grounding_gate_eval import _metrics  # reuse metric computation

THRESHOLD = float(os.environ.get("DTL_GROUND_THRESHOLD", "0.59"))
CORPUS = os.environ.get("DTL_CORPUS_GLOB", os.path.expanduser("~/.hermes/memory-sessions/dt-loci-*/findings.jsonl"))
MODEL = Path(__file__).resolve().parent.parent / "deep_think_loci/grounding/grounding_bleed_clf.joblib"

# Built-in focus queries for the example dama-gotchi targets; override with
# $DTL_TARGET_FOCUS (inline JSON or a path). Unknown targets fall back to the
# target name as the query, so the eval generalizes to any corpus.
DEFAULT_FOCUS = {
    "rooted-canary": "training/rooted_canary_e2e.py rooted-device telemetry canary + DGC-26 gate",
    "governance-gate": "training/governance_gate.py DGC checks, shadow-eval, fail-closed logic",
    "telemetry-ingest": "realtime ingest + gotchi_mqtt_bridge.py telemetry path and schema",
    "ant-training": "training/ant_trainer_base.py candidate export, publish, shadow-eval hook",
    "sensor-fusion": "android EskfFusion + sensors/tdoa_triangulation.py fusion correctness",
}


def _focus_map():
    raw = os.environ.get("DTL_TARGET_FOCUS")
    if raw:
        try:
            return json.loads(raw if raw.strip().startswith("{") else Path(raw).read_text())
        except Exception:
            pass
    return DEFAULT_FOCUS


def _target(rec):
    return next((t.split(":", 1)[1] for t in (rec.get("tags") or []) if t.startswith("dt_target:")), "")


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def run():
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not harness.OLLAMA_URL:
        print("[gate-qf-eval] OLLAMA_URL unset — this eval needs embeddings; skipping.")
        return

    focus = _focus_map()
    findings = []
    for f in glob.glob(CORPUS):
        for line in open(f, errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = _target(r)
            if t:
                findings.append({"text": (r.get("text") or "")[:2000], "target": t})
    targets = sorted({x["target"] for x in findings})
    if not findings or not targets:
        print(f"[gate-qf-eval] no targeted findings in {CORPUS}")
        return

    harness.ensure_collection() if not harness.DRY_RUN else None
    qvec = {t: _unit(harness.embed(focus.get(t, t))) for t in targets}
    texts = sorted({x["text"] for x in findings})
    fvec = {tx: _unit(harness.embed(tx)) for tx in texts}

    labels, cosv, feats = [], [], []
    for t in targets:
        q = qvec[t]
        for x in findings:
            cv = fvec[x["text"]]
            cos = sum(a * b for a, b in zip(q, cv))
            labels.append(1 if x["target"] == t else 0)
            cosv.append(cos)
            feats.append((q, cv, cos))

    print(f"[gate-qf-eval] run_date={run_date} targets={len(targets)} findings={len(findings)} "
          f"pairs={len(labels)} (pos={sum(labels)}) thr={THRESHOLD} dry_run={harness.DRY_RUN}")
    # Report each scorer at a FIXED threshold AND at its own best-F1 operating
    # point, so the model-vs-cosine F1 gap isn't just a threshold artifact
    # (cosine@0.59 vs model@0.5 is not a fair head-to-head). AUC is threshold-free.
    def _best_f1(scores):
        pos_tot = sum(labels) or 1
        best_thr, best_f1 = THRESHOLD, -1.0
        for t in sorted(set(scores)):
            tp = sum(1 for l, s in zip(labels, scores) if l == 1 and s >= t)
            fp = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= t)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / pos_tot
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 > best_f1:
                best_f1, best_thr = f1, t
        m = _metrics(labels, scores, best_thr)
        m["thr"] = best_thr
        return m

    cos_m = _metrics(labels, cosv, THRESHOLD)
    cos_best = _best_f1(cosv)
    print("  COSINE @%.2f   :" % THRESHOLD, {k: round(v, 3) for k, v in cos_m.items()})
    print("  COSINE @best=%.3f:" % cos_best["thr"], {k: round(v, 3) for k, v in cos_best.items() if k != "thr"})

    mdl_best = None
    if MODEL.exists():
        try:
            import joblib
            import numpy as np
            clf = joblib.load(str(MODEL))
            X = np.array([np.concatenate([np.abs(np.array(q) - np.array(cv)), np.array(q) * np.array(cv), [cos]])
                          for q, cv, cos in feats])
            proba = clf.predict_proba(X)[:, 1].tolist()
            mdl_m = _metrics(labels, proba, 0.5)
            mdl_best = _best_f1(proba)
            print("  MODEL  @0.50  :", {k: round(v, 3) for k, v in mdl_m.items()})
            print("  MODEL  @best=%.3f:" % mdl_best["thr"], {k: round(v, 3) for k, v in mdl_best.items() if k != "thr"})
        except Exception as e:
            print("  [model skipped]", e)

    if mdl_best:
        # Compare at each scorer's best F1 threshold, judged on F1 + AUC. Accuracy
        # is omitted from the criterion: pairs are ~4:1 bleed-imbalanced, so it is
        # inflated (predict-all-bleed already scores high).
        better = mdl_best["f1"] > cos_best["f1"] and mdl_best["auc"] >= cos_best["auc"]
        print(f"  IN-SAMPLE: model {'>' if better else '<='} cosine at best-F1 "
              f"(f1 {mdl_best['f1']:.3f} vs {cos_best['f1']:.3f}; auc {mdl_best['auc']:.3f} vs {cos_best['auc']:.3f})")
        print("  NOTE: in-sample only — the model is fit on THIS corpus, so these are optimistic and "
              "NOT a swap decision. The gate default is governed by the out-of-sample "
              "grounding_gate_oos_eval.py (leave-one-run-out).")

    if not harness.DRY_RUN:
        vec = harness.embed("deep_think_loci grounding gate query->finding eval")
        for tag, m in (("cosine", cos_m), ("model", mdl_m)):
            if not m:
                continue
            for name, val in m.items():
                if val == val:
                    harness.upsert_score(
                        task_id=f"dtl.gate_qf.{tag}.{name}",
                        task_name=f"gate query->finding {tag} {name}",
                        category="deep_think_loci",
                        score=float(val),
                        run_date=run_date,
                        context_preview=f"{len(findings)} findings x {len(targets)} targets",
                        vector=vec,
                    )
        print(f"[gate-qf-eval] persisted to {harness.EVAL_COLLECTION}")


if __name__ == "__main__":
    run()
