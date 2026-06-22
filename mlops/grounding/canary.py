#!/usr/bin/env python3
"""mlops/grounding/canary.py — grounding gate canary / promotion gate.

Reads investigation run findings from ~/.hermes/memory-sessions/dt-loci-*/findings.jsonl,
evaluates cosine gate vs a candidate classifier, and decides PROMOTE or HOLD.
Also computes Z-score drift against rolling history (the SLO-breach analog).

Importable as a module or run as a CLI:
  python3 mlops/grounding/canary.py \
      --candidate mlops/grounding/candidate.joblib \
      --target deep_think_loci/grounding/grounding_bleed_clf.joblib \
      --findings "~/.hermes/memory-sessions/dt-loci-*/findings.jsonl" \
      [--ollama http://localhost:11434] \
      [--min-margin 0.02] \
      [--dry-run]

Exit 0 = success (HOLD or PROMOTE). Exit 1 = drift detected.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORY_PATH = Path(__file__).resolve().parent / "canary_history.jsonl"
_PROMOTIONS_PATH = Path(__file__).resolve().parent / "promotions.jsonl"

EMB_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
DEFAULT_FINDINGS_GLOB = os.path.expanduser("~/.hermes/memory-sessions/dt-loci-*/findings.jsonl")
DEFAULT_OLLAMA = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
DEFAULT_THRESHOLD = float(os.environ.get("DTL_GROUND_THRESHOLD", "0.59"))
DEFAULT_MIN_MARGIN = 0.02
HISTORY_WINDOW = 10
MIN_FINDINGS_PER_RUN = 10


# ---------------------------------------------------------------------------
# Embedding — mirrors deep_think_loci/grounding/ground_gate.py exactly
# ---------------------------------------------------------------------------

def _embed(texts, ollama_url):
    url = ollama_url.rstrip("/") + "/v1/embeddings"
    out = []
    for i in range(0, len(texts), 16):
        body = json.dumps({"model": EMB_MODEL, "input": [t[:2000] for t in texts[i:i + 16]]}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        out += [d["embedding"] for d in json.loads(urllib.request.urlopen(req, timeout=60).read())["data"]]
    v = np.array(out, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------------------
# Per-run metric computation
# ---------------------------------------------------------------------------

def _dt_target(finding):
    tags = finding.get("tags") or []
    return {t.split(":", 1)[0]: t.split(":", 1)[1] for t in tags if ":" in t}.get("dt_target")


def _load_runs(findings_glob):
    runs = {}
    for path in glob.glob(os.path.expanduser(findings_glob)):
        run_id = Path(path).parent.name
        findings = []
        with open(path, errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                t = _dt_target(r)
                if t:
                    findings.append({"text": (r.get("text") or "")[:2000], "target": t})
        if len(findings) >= MIN_FINDINGS_PER_RUN:
            runs[run_id] = findings
    return runs


def _metrics(labels, scores, threshold):
    keep = [s >= threshold for s in scores]
    tp = sum(1 for l, k in zip(labels, keep) if l == 1 and k)
    fp = sum(1 for l, k in zip(labels, keep) if l == 0 and k)
    fn = sum(1 for l, k in zip(labels, keep) if l == 1 and not k)
    tn = sum(1 for l, k in zip(labels, keep) if l == 0 and not k)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def _eval_run(findings, threshold, clf, ollama_url):
    """
    Label construction: same dt_target = 1 (grounded), different target = 0 (bleed).
    For each finding, the query is the target name; cosine sim of embed(target) vs
    embed(finding) is the baseline signal.  We enumerate all (target, finding) pairs
    across the run's full target set.
    """
    texts = sorted({f["text"] for f in findings})
    targets = sorted({f["target"] for f in findings})

    text_vecs = _embed(texts, ollama_url)
    target_vecs = _embed(targets, ollama_url)

    text_idx = {t: i for i, t in enumerate(texts)}
    target_idx = {t: i for i, t in enumerate(targets)}

    labels, cos_scores, model_scores = [], [], []

    for target in targets:
        qv = target_vecs[target_idx[target]]
        for f in findings:
            fv = text_vecs[text_idx[f["text"]]]
            cos = float(np.dot(qv, fv))
            label = 1 if f["target"] == target else 0
            labels.append(label)
            cos_scores.append(cos)
            if clf is not None:
                feat = np.concatenate([np.abs(fv - qv), fv * qv, [cos]])
                model_scores.append(feat)

    if not labels or sum(labels) == 0:
        return None

    cos_m = _metrics(labels, cos_scores, threshold)
    if clf is not None and model_scores:
        import joblib as _joblib  # noqa: F401 — already imported at call site
        X = np.array(model_scores)
        proba = clf.predict_proba(X)[:, 1].tolist()
        model_m = _metrics(labels, proba, 0.5)
    else:
        model_m = None

    return {"n_pairs": len(labels), "cosine": cos_m, "model": model_m}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_gate(
    findings_glob=DEFAULT_FINDINGS_GLOB,
    candidate_model_path=None,
    threshold=DEFAULT_THRESHOLD,
    ollama_url=DEFAULT_OLLAMA,
    min_beat_margin=DEFAULT_MIN_MARGIN,
):
    """Evaluate cosine gate vs candidate model across all runs in findings_glob.

    Returns a result dict with aggregate metrics and a PROMOTE/HOLD decision.
    """
    import joblib

    clf = joblib.load(candidate_model_path) if candidate_model_path else None

    runs = _load_runs(findings_glob)
    if not runs:
        return {
            "n_runs": 0,
            "n_findings": 0,
            "cosine": {"mean_f1": float("nan"), "std_f1": float("nan"), "per_run": []},
            "model": {"mean_f1": float("nan"), "std_f1": float("nan"), "per_run": []},
            "beat_baseline": False,
            "delta_f1": float("nan"),
            "min_beat_margin": min_beat_margin,
            "decision": "HOLD",
        }

    cos_f1s, model_f1s = [], []
    cos_per_run, model_per_run = [], []
    n_findings_total = 0

    for run_id, findings in sorted(runs.items()):
        n_findings_total += len(findings)
        result = _eval_run(findings, threshold, clf, ollama_url)
        if result is None:
            continue
        cos_f1s.append(result["cosine"]["f1"])
        cos_per_run.append({"run_id": run_id, **result["cosine"]})
        if result["model"] is not None:
            model_f1s.append(result["model"]["f1"])
            model_per_run.append({"run_id": run_id, **result["model"]})

    def _agg(f1s, per_run):
        if not f1s:
            return {"mean_f1": float("nan"), "std_f1": float("nan"), "per_run": per_run}
        return {
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "per_run": per_run,
        }

    cos_agg = _agg(cos_f1s, cos_per_run)
    model_agg = _agg(model_f1s, model_per_run)

    cos_mean = cos_agg["mean_f1"]
    model_mean = model_agg["mean_f1"]

    if np.isnan(model_mean) or np.isnan(cos_mean):
        beat = False
        delta = float("nan")
    else:
        delta = model_mean - cos_mean
        beat = delta >= min_beat_margin

    decision = "PROMOTE" if beat else "HOLD"

    return {
        "n_runs": len(runs),
        "n_findings": n_findings_total,
        "cosine": cos_agg,
        "model": model_agg,
        "beat_baseline": beat,
        "delta_f1": delta,
        "min_beat_margin": min_beat_margin,
        "decision": decision,
    }


def promote(candidate_path, target_path, metrics):
    """Copy candidate to target and record the promotion in promotions.jsonl."""
    candidate_path = str(candidate_path)
    target_path = str(target_path)

    prev_sha = None
    if os.path.exists(target_path):
        with open(target_path, "rb") as fh:
            prev_sha = hashlib.sha256(fh.read()).hexdigest()

    shutil.copy2(candidate_path, target_path)

    record = {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "model_path": target_path,
        "metrics": metrics,
        "previous_sha256": prev_sha,
    }
    with open(_PROMOTIONS_PATH, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def zscore_drift_check(current_metrics, history_path=None):
    """Compare current run metrics against rolling history; flag SLO drift.

    current_metrics: dict with keys "cosine_f1" and "model_f1" (floats).
    history_path: path to canary_history.jsonl (defaults to the sibling file).
    Returns {"cosine_z": float, "model_z": float, "drift": bool, "reason": str}.
    """
    if history_path is None:
        history_path = _HISTORY_PATH

    history = []
    if os.path.exists(history_path):
        with open(history_path, errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    history.append(json.loads(line))
                except Exception:
                    continue

    recent = history[-HISTORY_WINDOW:] if len(history) >= HISTORY_WINDOW else history

    def _zscore(current_val, historical_vals):
        if len(historical_vals) < 2 or np.isnan(current_val):
            return float("nan")
        arr = np.array([v for v in historical_vals if not np.isnan(v)], dtype=np.float64)
        if len(arr) < 2:
            return float("nan")
        mu = np.mean(arr)
        sigma = np.std(arr, ddof=1)
        if sigma < 1e-9:
            return 0.0
        return float((current_val - mu) / sigma)

    hist_cos = [r.get("cosine_f1", float("nan")) for r in recent]
    hist_model = [r.get("model_f1", float("nan")) for r in recent]

    cos_z = _zscore(current_metrics.get("cosine_f1", float("nan")), hist_cos)
    model_z = _zscore(current_metrics.get("model_f1", float("nan")), hist_model)

    drift = False
    reasons = []
    # Negative Z below -2 means the current value dropped more than 2 SD below baseline.
    if not np.isnan(cos_z) and cos_z < -2.0:
        drift = True
        reasons.append(f"cosine_f1 dropped {cos_z:.2f}σ below recent baseline")
    if not np.isnan(model_z) and model_z < -2.0:
        drift = True
        reasons.append(f"model_f1 dropped {model_z:.2f}σ below recent baseline")

    return {
        "cosine_z": cos_z,
        "model_z": model_z,
        "drift": drift,
        "reason": "; ".join(reasons) if reasons else "ok",
    }


def _append_history(record):
    with open(_HISTORY_PATH, "a") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Grounding gate canary: evaluate candidate model vs cosine baseline and optionally promote."
    )
    ap.add_argument("--candidate", required=True, help="Path to candidate.joblib")
    ap.add_argument("--target", required=True,
                    help="Path to the live joblib (promote target)")
    ap.add_argument("--findings", default=DEFAULT_FINDINGS_GLOB,
                    help="Glob for findings.jsonl files")
    ap.add_argument("--ollama", default=DEFAULT_OLLAMA,
                    help="Ollama base URL (no /v1 suffix)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="Cosine threshold for the baseline gate")
    ap.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN,
                    help="Minimum F1 delta over cosine required to promote")
    ap.add_argument("--dry-run", action="store_true",
                    help="Evaluate and report but do not promote or write history")
    a = ap.parse_args()

    print(f"[canary] evaluating candidate={a.candidate} findings={a.findings}")
    result = evaluate_gate(
        findings_glob=a.findings,
        candidate_model_path=a.candidate,
        threshold=a.threshold,
        ollama_url=a.ollama,
        min_beat_margin=a.min_margin,
    )

    cos_f1 = result["cosine"]["mean_f1"]
    model_f1 = result["model"]["mean_f1"]
    delta = result["delta_f1"]

    print(f"[canary] n_runs={result['n_runs']} n_findings={result['n_findings']}")
    print(f"[canary] cosine mean_f1={cos_f1:.4f} std={result['cosine']['std_f1']:.4f}")
    print(f"[canary] model  mean_f1={model_f1:.4f} std={result['model']['std_f1']:.4f}")
    print(f"[canary] delta={delta:+.4f} min_margin={a.min_margin} decision={result['decision']}")

    current_metrics = {"cosine_f1": cos_f1, "model_f1": model_f1}
    drift_result = zscore_drift_check(current_metrics)
    print(f"[canary] drift_check cosine_z={drift_result['cosine_z']:.2f} "
          f"model_z={drift_result['model_z']:.2f} drift={drift_result['drift']} "
          f"reason={drift_result['reason']}")

    history_record = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_runs": result["n_runs"],
        "cosine_f1": cos_f1,
        "model_f1": model_f1,
        "decision": result["decision"],
    }

    if not a.dry_run:
        _append_history(history_record)
        if result["decision"] == "PROMOTE":
            promote(a.candidate, a.target, result)
            print(f"[canary] PROMOTED {a.candidate} -> {a.target}")
        else:
            print("[canary] HOLD — candidate did not beat cosine by the required margin")
    else:
        print("[canary] dry-run: no files written")

    if drift_result["drift"]:
        print(f"[canary] ALERT: SLO drift detected — {drift_result['reason']}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
