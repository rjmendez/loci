"""
Faithful query->finding eval for the deep-think-loci grounding gate.

The deployed gate operates query->finding (a target's focus query vs each
candidate finding), NOT finding<->finding (which is the classifier's training
task, measured by grounding_gate_eval.py). This measures the gate's ACTUAL
operation on the corpus and compares the cosine threshold against the trained
classifier — the eval that decides whether to swap the trained model in as the
gate default.

For every (target focus query, finding) pair: cosine = sim(embed(focus),
embed(finding)); label = 1 if the finding belongs to that target else 0 (bleed).
Reports recall / bleed_rejection / f1 / accuracy / auc for the cosine threshold
and (if present) the trained model, plus a swap verdict.

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
    cos_m = _metrics(labels, cosv, THRESHOLD)
    print("  COSINE @%.2f :" % THRESHOLD, {k: round(v, 3) for k, v in cos_m.items()})

    mdl_m = None
    if MODEL.exists():
        try:
            import joblib
            import numpy as np
            clf = joblib.load(str(MODEL))
            X = np.array([np.concatenate([np.abs(np.array(q) - np.array(cv)), np.array(q) * np.array(cv), [cos]])
                          for q, cv, cos in feats])
            proba = clf.predict_proba(X)[:, 1].tolist()
            mdl_m = _metrics(labels, proba, 0.5)
            print("  MODEL  @0.5 :", {k: round(v, 3) for k, v in mdl_m.items()})
        except Exception as e:
            print("  [model skipped]", e)

    if mdl_m:
        better = mdl_m["f1"] > cos_m["f1"] and mdl_m["accuracy"] >= cos_m["accuracy"]
        print(f"  VERDICT: trained model {'BEATS' if better else 'does NOT beat'} cosine on query->finding "
              f"(f1 {mdl_m['f1']:.3f} vs {cos_m['f1']:.3f}, acc {mdl_m['accuracy']:.3f} vs {cos_m['accuracy']:.3f})"
              f" -> {'SWAP IN the trained model' if better else 'keep cosine default'}")

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
