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


def _record_pair(rec: dict) -> tuple[str, str]:
    """(claim, evidence) for one dataset row.

    grounding_dataset.jsonl rows are {claim, evidence, label, signal, cos} — none
    of text/content/query exists on any of them, so the old field chain measured
    every row as too short and dropped the whole corpus before scoring.
    """
    claim = (rec.get("text") or rec.get("claim") or rec.get("content")
             or rec.get("query", ""))
    evidence = rec.get("evidence") or rec.get("context", "")
    return claim, evidence


def _feature_contract():
    """deep_think_loci/grounding/features.py — the one grounding feature contract."""
    grounding = Path(__file__).resolve().parents[2] / "deep_think_loci" / "grounding"
    if str(grounding) not in sys.path:
        sys.path.insert(0, str(grounding))
    import features
    return features


def _say(msg: str) -> None:
    """Print a "the lane is dark, and here is why" line where the nightly can see it.

    mlops/loop.py runs this script through _run(), which drains the child's
    stdout with echo=True and its stderr with echo=False, and _run_active_learn
    returns only the exit code. A diagnostic on stderr is therefore invisible in
    the one log that reads it -- the nightly would print "boundary=0" alone,
    which is the defect this file was changed to fix.
    """
    print(f"[active_learn] {msg}", flush=True)


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
        import numpy as np
        _feat = _feature_contract()
        clf = joblib.load(model_path)
    except Exception as exc:
        _say(f"could not load model: {exc}")
        return []

    records = _load_dataset(dataset_path)
    if not records:
        return []

    # The live classifier scores (claim, evidence) PAIRS at the width it was
    # trained on. This handed it one raw 768-d embedding, so predict_proba raised
    # on every record and the per-record `except: continue` ate it — a field-name
    # fix alone would still have scored nothing, just slower.
    dim = int(getattr(clf, "n_features_in_", _feat.LEGACY_DIM))
    if dim not in _feat.supported_dims():
        _say(f"{model_path} expects {dim} features; this sampler builds "
             f"{_feat.supported_dims()} — refusing to score with a model whose "
             "feature contract is unknown.")
        return []

    usable = [(rec, c, e) for rec, (c, e) in ((r, _record_pair(r)) for r in records)
              if len(c) >= 20 and e]
    if not usable:
        _say(f"examined {len(records)} rows, 0 carried a (claim, evidence) "
             "pair this model can score")
        return []

    # One embed per distinct string, not two per row: the corpus repeats its
    # claims and evidences heavily (5418 rows over 143 of each).
    cache: dict[str, list] = {}
    failures = []
    for _, claim, evidence in usable:
        for text in (claim, evidence):
            if text in cache:
                continue
            try:
                cache[text] = _embed(text, ollama_url, embed_model)
            except Exception as exc:
                failures.append(exc)
                cache[text] = None
    if failures:
        # One line, not one per string: with Ollama down this is every distinct
        # text in the corpus and the nightly log is the only reader.
        _say(f"{len(failures)}/{len(cache)} embeds failed, first: {failures[0]}")

    rows = [(rec, c, e) for rec, c, e in usable if cache.get(c) and cache.get(e)]
    if not rows:
        _say(f"{len(usable)} scorable rows, 0 embedded")
        return []

    def _unit(texts):
        arr = np.asarray([cache[t] for t in texts], dtype=np.float32)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

    claims = [c for _, c, _ in rows]
    evidences = [e for _, _, e in rows]
    try:
        feats = _feat.make_features(claims, evidences, _unit(claims), _unit(evidences),
                                    dim=dim)
        probas = clf.predict_proba(feats)[:, 1]
    except Exception as exc:
        _say(f"scoring {len(rows)} rows failed: {exc}")
        return []

    scored = [{"rec": rec, "proba": float(p), "uncertainty": float(abs(p - 0.5))}
              for (rec, _, _), p in zip(rows, probas)]

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
    # NOT fixed alongside boundary_samples, deliberately. This reads the same
    # absent text/content fields and so returns [] on the live corpus, but
    # teaching it `claim` would emit claim-vs-claim pairs under {text, context}
    # keys — neither the negative this (claim, evidence) classifier needs nor a
    # shape the dataset can ingest. Fixing it is a design decision about what a
    # hard negative is here, not a field-name repair.
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
