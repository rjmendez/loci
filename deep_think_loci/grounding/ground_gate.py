#!/usr/bin/env python3
"""Grounding gate — drop RAG-bleed evidence before a model reasons over it.

Given a query (the topic/claim being grounded) and candidate findings, keep only
those whose nomic-embedding cosine to the query clears a threshold (~0.65), which
filters the cross-target similarity-bleed (cosine 0.35–0.55) that produced the
v1/v2 hallucinations. Runs locally against Ollama — ~$0 marginal, no extra agent.

v0 is a cosine threshold (competitive on current data). A trained pair-feature
classifier (grounding_bleed_clf.joblib) can be dropped in via --model once it
beats cosine on a larger corpus.

Usage:
  python3 ground_gate.py --query "<topic/focus>" [--threshold 0.65] [--model clf.joblib] \
      --in candidates.json [--out kept.json]
  # candidates.json: [{"id": "...", "text": "..."}, ...]  (or {"findings": [...]})
  # also reads candidates from stdin if --in is omitted.
Exit: prints JSON {kept:[...], dropped:[...], threshold, mode} to --out or stdout.
"""
import argparse, json, os, sys, urllib.request
import numpy as np

# Loci convention: OLLAMA_BASE_URL has no /v1 suffix; EMBED_MODEL names the embedder.
_BASE = (os.environ.get("OLLAMA_BASE_URL") or "http://100.73.200.19:11434").rstrip("/")
OLLAMA = _BASE + "/v1/embeddings"
EMB_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")


def embed(texts):
    out = []
    for i in range(0, len(texts), 16):
        body = json.dumps({"model": EMB_MODEL, "input": [t[:2000] for t in texts[i:i + 16]]}).encode()
        req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
        out += [d["embedding"] for d in json.loads(urllib.request.urlopen(req, timeout=60).read())["data"]]
    v = np.array(out, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def gate(query, cands, threshold=0.59, model=None):
    """Filter candidate findings against a query. Returns the result dict.

    cands: list of {"id","text"} dicts (or bare strings). Pure logic + embed();
    patch embed() to unit-test without a network call.
    """
    cands = [c if isinstance(c, dict) else {"text": str(c)} for c in cands]
    if not cands:
        return {"kept": [], "dropped": [], "threshold": threshold, "mode": "empty",
                "n_in": 0, "n_kept": 0, "n_dropped": 0}

    qv = embed([query])[0]
    cv = embed([c.get("text", "") for c in cands])
    cos = (cv @ qv).tolist()

    if model:
        import joblib
        clf = joblib.load(model)
        feats = np.array([np.concatenate([np.abs(cv[i] - qv), cv[i] * qv, [cos[i]]]) for i in range(len(cands))])
        score = clf.predict_proba(feats)[:, 1].tolist()
        keep = [s >= 0.5 for s in score]
        mode = "model:" + model.split("/")[-1]
    else:
        score = cos
        keep = [c >= threshold for c in cos]
        mode = f"cosine>={threshold}"

    kept, dropped = [], []
    for c, cs, sc, k in zip(cands, cos, score, keep):
        rec = {**c, "cos": round(float(cs), 3), "score": round(float(sc), 3)}
        (kept if k else dropped).append(rec)
    return {"kept": kept, "dropped": dropped, "threshold": threshold, "mode": mode,
            "n_in": len(cands), "n_kept": len(kept), "n_dropped": len(dropped)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--threshold", type=float, default=0.59)
    ap.add_argument("--model", default=None, help="optional joblib pair-feature classifier (drop-in upgrade)")
    ap.add_argument("--in", dest="infile", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    raw = json.load(open(a.infile)) if a.infile else json.load(sys.stdin)
    cands = raw.get("findings", raw) if isinstance(raw, dict) else raw
    result = gate(a.query, cands, a.threshold, a.model)
    out = json.dumps(result, indent=1)
    (open(a.out, "w").write(out + "\n")) if a.out else print(out)


if __name__ == "__main__":
    main()
