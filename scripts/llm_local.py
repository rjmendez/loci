#!/usr/bin/env python3
"""CLI wrapper for llm_local.generate() — one-shot local (Ollama) generation.

Prints the generated text to stdout. If generation failed (ok=False) it still exits
non-zero and writes a short note to stderr, so shell callers can branch on $? and fall
back (e.g. to Claude Haiku, per the shared gen_fn contract). Fail-open: never raises.

Usage:
    scripts/llm_local.py 'summarize this in one line'      # prompt from argv
    echo 'summarize this' | scripts/llm_local.py           # prompt on stdin
    scripts/llm_local.py --json '{"as":"json"}' 'give me JSON'   # request+validate JSON

Options:
    --json / --format json   request JSON output and validate it parses
    --model <tag>            override model (default qwen2.5:3b)
    --max-tokens <n>         override num_predict (default 256)

Reads OLLAMA_BASE_URL / OLLAMA_URL from env (same as mcp/embed_ops.py). See mcp/llm_local.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))


def main() -> int:
    import llm_local

    fmt = None
    model = "qwen2.5:3b"
    max_tokens = 256
    args = []
    it = iter(sys.argv[1:])
    for a in it:
        if a in ("--json",) or a == "--format":
            if a == "--format":
                next(it, None)  # consume the 'json' value
            fmt = "json"
        elif a == "--model":
            model = next(it, model)
        elif a == "--max-tokens":
            try:
                max_tokens = int(next(it, max_tokens))
            except (TypeError, ValueError):
                pass
        else:
            args.append(a)

    prompt = " ".join(args).strip() if args else sys.stdin.read().strip()
    if not prompt:
        print("error: empty prompt (pass as argv or on stdin)", file=sys.stderr)
        return 2

    res = llm_local.generate(prompt, model=model, fmt=fmt, max_tokens=max_tokens)
    if res.get("text"):
        print(res["text"])
    if not res.get("ok"):
        print(f"note: generation not ok (model={res.get('model')}, fmt={fmt})",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
