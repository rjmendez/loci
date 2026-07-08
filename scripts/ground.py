#!/usr/bin/env python3
"""CLI wrapper for grounding.ground() — produce a grounding block for a task.

The main loop runs this BEFORE spawning a workflow, then passes the block as the
workflow's ``args.ground`` so every fan-out agent starts with relevant prior context
(structured-first: named cases + entities + code graph + semantic RAG, noise-filtered,
char-budgeted). See mcp/grounding.py.

Usage:
    scripts/ground.py '{"title":"...","focus":"...","caseIds":["INV-1"],"entities":["1.2.3.4"]}' [budgetChars]
    echo '<task-json>' | scripts/ground.py            # task on stdin

Reads config from the loci MCP env (OLLAMA_BASE_URL/QDRANT_URL/QDRANT_API_KEY) if set;
degrades gracefully (structured/memory-only) when a source is unavailable.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))


def main() -> int:
    raw = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] not in ("-", "") else sys.stdin.read()
    try:
        task = json.loads(raw)
    except Exception as exc:
        print(f"error: task must be JSON ({exc})", file=sys.stderr)
        return 2
    opts = {}
    if len(sys.argv) > 2:
        try:
            opts["budgetChars"] = int(sys.argv[2])
        except ValueError:
            pass
    import grounding  # lazy: grounding.py lazy-imports server, so this is cheap
    result = grounding.ground(task, opts)
    # print the block to stdout (empty if nothing grounded); provenance/degraded to stderr
    print(result["block"])
    print(f"[ground: {result['chars']} chars, sources={result['sources']}, degraded={result['degraded']}]",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
