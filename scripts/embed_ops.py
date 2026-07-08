#!/usr/bin/env python3
"""CLI for embed_ops — cheap semantic ops on the local-GPU embedding path.

Usage:
  embed_ops.py dedup '<json list of items>' [threshold] [text_key]
      -> {clusters, kept, dropped, degraded}   (near-duplicate clustering)
  embed_ops.py relevance '<topic>' '<json list of texts>'
      -> {scores, degraded}                    (cosine of each text to topic)

Reads OLLAMA_BASE_URL / EMBED_MODEL from the environment (same as the Loci server).
Fail-open: with no embeddings, dedup drops nothing and relevance returns null scores.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    import embed_ops as E
    op = sys.argv[1]
    if op == "dedup":
        items = json.loads(sys.argv[2])
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.88
        key = sys.argv[4] if len(sys.argv) > 4 else None
        out = E.dedup(items, threshold=threshold, key=key)
    elif op == "relevance":
        topic = sys.argv[2]
        texts = json.loads(sys.argv[3]) if len(sys.argv) > 3 else []
        out = E.relevance(texts, topic)
    else:
        print(f"unknown op: {op}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2))
    if out.get("degraded"):
        print("[embed_ops: DEGRADED — embeddings unavailable, result is a no-op fallback]",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
