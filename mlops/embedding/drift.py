#!/usr/bin/env python3
"""mlops/embedding/drift.py — Embedding drift detection vs. an anchor snapshot.

Implements the LMEB benchmark pattern (arXiv:2603.12572, Mar 2026): sample a
representative set of texts, embed them, save as anchor.npz. On subsequent runs,
re-embed the same texts and measure cosine similarity drift. Exit 1 if drift
exceeds threshold, triggering loop.py to emit a contrastive fine-tune script.

Usage:
    # Build anchor
    python3 mlops/embedding/drift.py --dataset grounding_dataset.jsonl \
        --ollama http://localhost:11434 --anchor mlops/embedding/anchor.npz --build-anchor

    # Measure drift
    python3 mlops/embedding/drift.py --dataset grounding_dataset.jsonl \
        --ollama http://localhost:11434 --anchor mlops/embedding/anchor.npz
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

DEFAULT_ANCHOR = str(Path(__file__).parent / "anchor.npz")
DEFAULT_OLLAMA = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
DEFAULT_N = 100
DRIFT_THRESHOLD = 0.02


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


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _sample_texts(dataset_path: str, n: int) -> list[str]:
    import random
    texts = []
    try:
        with open(dataset_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line.strip())
                    t = rec.get("text") or rec.get("content") or rec.get("query", "")
                    if len(t) > 30:
                        texts.append(t)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    random.seed(42)
    return random.sample(texts, min(n, len(texts)))


def build_anchor(
    dataset_path: str,
    ollama_url: str = DEFAULT_OLLAMA,
    model: str = DEFAULT_MODEL,
    n: int = DEFAULT_N,
    anchor_path: str = DEFAULT_ANCHOR,
) -> dict:
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy not installed"}

    texts = _sample_texts(dataset_path, n)
    if not texts:
        return {"error": "no texts found in dataset"}

    embeddings = []
    for i, text in enumerate(texts):
        try:
            vec = _embed(text, ollama_url, model)
            embeddings.append(vec)
        except Exception as exc:
            print(f"[drift] embed failed for sample {i}: {exc}", file=sys.stderr)
            continue

    if not embeddings:
        return {"error": "all embeddings failed"}

    arr = np.array(embeddings, dtype=np.float32)
    Path(anchor_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(anchor_path, embeddings=arr, texts=texts[:len(embeddings)])
    print(f"[drift] anchor built: {len(embeddings)} embeddings → {anchor_path}")
    return {"n_anchored": len(embeddings), "anchor_path": anchor_path}


def measure_drift(
    anchor_path: str = DEFAULT_ANCHOR,
    ollama_url: str = DEFAULT_OLLAMA,
    model: str = DEFAULT_MODEL,
    threshold: float = DRIFT_THRESHOLD,
) -> dict:
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy not installed"}

    if not os.path.exists(anchor_path):
        return {"error": f"anchor not found: {anchor_path}"}

    data = np.load(anchor_path, allow_pickle=True)
    anchor_embs = data["embeddings"]
    texts = data["texts"].tolist()

    cosines = []
    n_drifted = 0
    for i, text in enumerate(texts):
        try:
            live_vec = _embed(str(text), ollama_url, model)
        except Exception:
            continue
        cos = _cosine(anchor_embs[i].tolist(), live_vec)
        cosines.append(cos)
        if cos < (1.0 - threshold):
            n_drifted += 1

    if not cosines:
        return {"error": "no embeddings produced"}

    mean_cos = sum(cosines) / len(cosines)
    drift_score = 1.0 - mean_cos
    return {
        "mean_cosine": mean_cos,
        "drift_score": drift_score,
        "n_drifted_095": n_drifted,
        "n_texts": len(cosines),
        "threshold": threshold,
        "exceeded": drift_score > threshold,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Embedding drift detection vs anchor snapshot")
    ap.add_argument("--dataset", required=True, help="Path to grounding_dataset.jsonl")
    ap.add_argument("--anchor", default=DEFAULT_ANCHOR)
    ap.add_argument("--ollama", default=DEFAULT_OLLAMA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="Anchor sample size")
    ap.add_argument("--threshold", type=float, default=DRIFT_THRESHOLD)
    ap.add_argument("--build-anchor", action="store_true", help="Build anchor instead of measuring")
    ap.add_argument("--out", default=None, help="Write drift JSON to this path")
    a = ap.parse_args()

    if a.build_anchor:
        result = build_anchor(a.dataset, a.ollama, a.model, a.n, a.anchor)
        if "error" in result:
            print(f"[drift] ERROR: {result['error']}", file=sys.stderr)
            sys.exit(1)
        print(f"[drift] anchor built: n={result['n_anchored']}")
        sys.exit(0)

    result = measure_drift(a.anchor, a.ollama, a.model, a.threshold)
    if "error" in result:
        print(f"[drift] ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"[drift] mean_cosine={result['mean_cosine']:.4f} drift_score={result['drift_score']:.4f} "
          f"n_drifted={result['n_drifted_095']}/{result['n_texts']} exceeded={result['exceeded']}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2))

    sys.exit(1 if result["exceeded"] else 0)


if __name__ == "__main__":
    main()
