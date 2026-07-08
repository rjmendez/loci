#!/usr/bin/env python3
"""CLI for text_ops — generation-tier basic ops on the local-GPU (Ollama) path.

Usage:
  text_ops.py classify '<text>' '<json list of labels>'
      -> {label, degraded}     (single best label from the set)
  text_ops.py compress '<text>' [max_chars]
      -> {text, degraded}      (semantic condense under a char budget)

Generation comes from llm_local.generate (Ollama qwen2.5:3b), pinned resident via
keep_alive by that module. Fail-open: with no / not-ok generation, classify returns
label=null and compress returns char-truncated text, both with degraded=true.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    import text_ops as T
    op = sys.argv[1]
    if op == "classify":
        text = sys.argv[2]
        labels = json.loads(sys.argv[3]) if len(sys.argv) > 3 else []
        out = T.classify(text, labels)
    elif op == "compress":
        text = sys.argv[2]
        max_chars = int(sys.argv[3]) if len(sys.argv) > 3 else 600
        out = T.compress(text, max_chars=max_chars)
    else:
        print(f"unknown op: {op}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2))
    if out.get("degraded"):
        print("[text_ops: DEGRADED — generation unavailable/out-of-set, result is a "
              "fail-open fallback]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
