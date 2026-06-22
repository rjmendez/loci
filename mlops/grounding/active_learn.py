#!/usr/bin/env python3
"""mlops/grounding/active_learn.py — Boundary sampling and hard negative synthesis.

Identifies the most uncertain examples for the live grounding classifier
(boundary samples where |proba - 0.5| is smallest) and synthesizes hard
negatives via vocabulary-overlap cross-pairing of positives.

Both techniques improve grounding calibration with minimal new data collection.
Written to active_candidates.jsonl for human review or auto-ingestion.

Usage:
    python3 mlops/grounding/active_learn.py \
        --model deep_think_loci/grounding/grounding_bleed_clf.joblib \
        --dataset deep_think_loci/grounding/grounding_dataset.jsonl \
        --out mlops/grounding/active_candidates.jsonl \
        --ollama http://localhost:11434
"""

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_OLLAMA = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
DEFAULT_N_BOUNDARY = 100
DEFAULT_N_HARD_NEG = 50
DEFAULT_BAND = 0.2
DEFAULT_EMB_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")


def _embed(text: str, ollama_url: str, model: str) -> list[float]:
    import urllib.request
    payload = json.dumps({"model": model, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{ollama_url}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["embedding"]


def _load_dataset(dataset_path: str) -> list[dict]:
    records = []
    try:
        with open(dataset_path) as fh:
            for line in fh:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return records


def boundary_samples(
    model_path: str,
    dataset_path: str,
    ollama_url: str = DEFAULT_OLLAMA,
    embed_model: str = DEFAULT_EMB_MODEL,
    n: int = DEFAULT_N_BOUNDARY,
    uncertainty_band: float = DEFAULT_BAND,
) -> list[dict]:
    if not os.path.exists(model_path):
        return []
    try:
        import joblib
        clf = joblib.load(model_path)
    except Exception as exc:
        print(f"[active_learn] could not load model: {exc}", file=sys.stderr)
        return []

    records = _load_dataset(dataset_path)
    if not records:
        return []

    scored = []
    for rec in records:
        text = rec.get("text") or rec.get("content") or rec.get("query", "")
        if len(text) < 20:
            continue
        try:
            vec = _embed(text, ollama_url, embed_model)
            proba = clf.predict_proba([vec])[0][1]
            uncertainty = abs(proba - 0.5)
            scored.append({"rec": rec, "proba": float(proba), "uncertainty": float(uncertainty)})
        except Exception:
            continue

    boundary = [s for s in scored if s["uncertainty"] <= uncertainty_band / 2]
    boundary.sort(key=lambda x: x["uncertainty"])
    candidates = []
    for s in boundary[:n]:
        entry = dict(s["rec"])
        entry["candidate_type"] = "boundary"
        entry["proba"] = s["proba"]
        entry["uncertainty"] = s["uncertainty"]
        candidates.append(entry)
    return candidates


def hard_negatives(
    dataset_path: str,
    n: int = DEFAULT_N_HARD_NEG,
) -> list[dict]:
    records = _load_dataset(dataset_path)
    positives = [r for r in records if r.get("label", 0) == 1]
    if len(positives) < 2:
        return []

    def overlap_score(a: str, b: str) -> int:
        return len(set(a.lower().split()) & set(b.lower().split()))

    negatives = []
    seen = set()
    for i, pos_a in enumerate(positives):
        text_a = pos_a.get("text") or pos_a.get("content", "")
        for j, pos_b in enumerate(positives):
            if i == j:
                continue
            pair_key = tuple(sorted([i, j]))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            text_b = pos_b.get("text") or pos_b.get("content", "")
            ov = overlap_score(text_a, text_b)
            if ov >= 3:
                negatives.append({
                    "text": text_a,
                    "context": text_b,
                    "label": 0,
                    "candidate_type": "hard_negative",
                    "vocab_overlap": ov,
                    "source_indices": [i, j],
                })
        if len(negatives) >= n * 3:
            break

    negatives.sort(key=lambda x: x["vocab_overlap"], reverse=True)
    return negatives[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Active learning: boundary samples + hard negatives")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ollama", default=DEFAULT_OLLAMA)
    ap.add_argument("--n-boundary", type=int, default=DEFAULT_N_BOUNDARY)
    ap.add_argument("--n-hard", type=int, default=DEFAULT_N_HARD_NEG)
    ap.add_argument("--band", type=float, default=DEFAULT_BAND)
    a = ap.parse_args()

    boundary = boundary_samples(a.model, a.dataset, a.ollama, n=a.n_boundary, uncertainty_band=a.band)
    hard = hard_negatives(a.dataset, n=a.n_hard)
    all_candidates = boundary + hard

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        for c in all_candidates:
            fh.write(json.dumps(c) + "\n")

    print(f"[active_learn] boundary={len(boundary)} hard_negatives={len(hard)} total={len(all_candidates)} → {a.out}")


if __name__ == "__main__":
    main()
