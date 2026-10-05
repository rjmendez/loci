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
from importlib import util
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))


def _load_real_module():
    """Load mcp/query_expand.py by path, under a private module name.

    This file is a CLI wrapper, but it shares a module NAME with mcp/query_expand.py,
    and scripts/judge_eval.py puts scripts/ AHEAD of mcp/ on sys.path, so the server's
    `import query_expand` can resolve here. The old re-export did
    `from query_expand import expand`, which in that case imported this half-initialised
    module itself, hit ImportError and left expand=None: _rag_expand_queries then failed
    open on every call and the measured +4% nDCG@10 from expansion was silently off.
    Loading by path (as scripts/llm_local.py does) cannot resolve back to this file.
    """
    path = Path(__file__).resolve().parent.parent / "mcp" / "query_expand.py"
    spec = util.spec_from_file_location("_loci_mcp_query_expand", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load {path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


expand = _load_real_module().expand  # re-export: whichever module wins the name works


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    query = sys.argv[1]
    n_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_keywords = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    out = expand(query, n_queries=n_queries, n_keywords=n_keywords)
    print(json.dumps(out, indent=2))
    if out.get("degraded"):
        print("[query_expand: DEGRADED — generation unavailable/unusable, "
              "returning original query only]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
