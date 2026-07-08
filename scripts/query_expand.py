#!/usr/bin/env python3
"""CLI for query_expand — RAG query expansion (HyDE-lite) on the local-GPU generation tier.

Usage:
  query_expand.py '<query>' [n_queries] [n_keywords]
      -> {queries, keywords, degraded}

Generation runs via llm_local.generate (Ollama qwen2.5:3b), imported lazily. Fail-open:
on any generation/parse failure the result is {queries:[query], keywords:[], degraded:true}
so retrieval always has at least the original query to run.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    import query_expand as Q
    query = sys.argv[1]
    n_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_keywords = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    out = Q.expand(query, n_queries=n_queries, n_keywords=n_keywords)
    print(json.dumps(out, indent=2))
    if out.get("degraded"):
        print("[query_expand: DEGRADED — generation unavailable/unusable, "
              "returning original query only]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
