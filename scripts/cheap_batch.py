#!/usr/bin/env python3
"""Resolve workflow tasks on the cheap tiers, so no Claude agent has to.

A Workflow `agent()` is always a Claude subagent — there is no hook that points
one at vLLM or OpenRouter. Offloading therefore cannot mean changing the model an
agent runs on; it means doing the work BEFORE the workflow and injecting the
answer, the way `loci-native.js` already does with `args.ground` and
`args.graphFacts` (a `tier:'graph'` task resolves from an injected fact with no
agent and zero tokens).

This produces facts for that same mechanism from the local and remote generation
tiers. Output is keyed exactly like `scripts/graph_facts.py`, so `loci-native.js`
consumes it unchanged:

    { "<key>": {"text": "...", "model": "...", "tier": "local|remote", "ok": true} }

Measured on the wf_6e13c6c2-8e0 run this exists to shrink: agent *returns* were
3.3% of subagent token spend. The other 96.7% was input — 283 Bash calls in a
single agent transcript, rediscovering the repo. A task whose answer can be
produced once, cheaply, and handed over is a task no agent needs to read for.

Usage:
    cheap_batch.py tasks.json > facts.json
    echo '[{"key":"k","prompt":"..."}]' | cheap_batch.py > facts.json

    # then
    Workflow({scriptPath: ".../loci-native.js",
              args: {ground: block, cheapFacts: facts, tasks: [
                  {id: "k", title: "...", tier: "cheap"}]}})

Tier order is local-first: vLLM and Ollama cost nothing per call and keep the
corpus on the box. --remote adds the OpenRouter ladder for whatever the local
tiers could not serve, which on a flaky GPU node is the difference between a
stalled batch and a slower one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "mcp"))

MAX_TOKENS = int(os.environ.get("LOCI_CHEAP_MAX_TOKENS", "512"))


def _load_tasks(argv_path: Optional[str]) -> list:
    raw = open(argv_path).read() if argv_path else sys.stdin.read()
    tasks = json.loads(raw)
    if isinstance(tasks, dict):
        tasks = [{"key": k, "prompt": v} for k, v in tasks.items()]
    out = []
    for t in tasks:
        key = str(t.get("key") or t.get("id") or "").strip()
        prompt = str(t.get("prompt") or t.get("focus") or "").strip()
        if key and prompt:
            out.append({"key": key, "prompt": prompt})
    return out


def _local_batch(prompts: list, fmt: Optional[str]) -> list:
    try:
        import batched_gen
    except Exception as exc:
        return [{"text": "", "ok": False, "why": f"local tier unavailable: {exc!r}"}
                for _ in prompts]
    # model=None on purpose: the batched and Ollama tiers name the same model
    # differently and an unrecognised name is rejected outright.
    return batched_gen.generate_batch(prompts, max_tokens=MAX_TOKENS, fmt=fmt)


def _remote_batch(prompts: list, fmt: Optional[str]) -> list:
    try:
        import openrouter
    except Exception as exc:
        return [{"text": "", "ok": False, "why": f"remote tier unavailable: {exc!r}"}
                for _ in prompts]
    if not openrouter.available():
        return [{"text": "", "ok": False, "why": "no OpenRouter key configured"}
                for _ in prompts]
    return openrouter.generate_batch(prompts, max_tokens=MAX_TOKENS, fmt=fmt)


def resolve(tasks: list, remote: bool = False, fmt: Optional[str] = None,
            local_fn: Optional[Callable] = None,
            remote_fn: Optional[Callable] = None) -> dict:
    """Answer every task on the cheapest tier that can, and say which one did.

    A task that no tier could serve is returned with ``ok: false`` rather than
    omitted — `loci-native.js` degrades a factless task to a mechanical agent, and
    it can only do that if it can see the gap.
    """
    if not tasks:
        return {}
    local_fn = local_fn or _local_batch
    remote_fn = remote_fn or _remote_batch
    prompts = [t["prompt"] for t in tasks]

    results = local_fn(prompts, fmt)
    facts = {}
    for t, r in zip(tasks, results or []):
        r = r or {}
        facts[t["key"]] = {
            "text": r.get("text", ""), "ok": bool(r.get("ok")),
            "tier": "local" if r.get("ok") else None,
            "model": r.get("model"),
            "why": r.get("why", ""),
        }

    if remote:
        todo = [t for t in tasks if not facts[t["key"]]["ok"]]
        if todo:
            rem = remote_fn([t["prompt"] for t in todo], fmt)
            for t, r in zip(todo, rem or []):
                r = r or {}
                if r.get("ok"):
                    facts[t["key"]] = {"text": r.get("text", ""), "ok": True,
                                       "tier": "remote", "model": r.get("model"),
                                       "why": ""}
                else:
                    facts[t["key"]]["why"] = r.get("why", "") or "remote tier declined"
    return facts


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tasks", nargs="?", help="JSON file; omit to read stdin")
    ap.add_argument("--remote", action="store_true",
                    help="fall through to OpenRouter for whatever the local tiers "
                         "could not serve. Off by default: findings carry internal "
                         "addressing, so leaving the box is an explicit choice.")
    ap.add_argument("--json-out", action="store_true",
                    help="ask the tiers for JSON and validate it")
    ap.add_argument("--stats", action="store_true", help="tier summary to stderr")
    args = ap.parse_args(argv)

    try:
        tasks = _load_tasks(args.tasks)
    except Exception as exc:
        print(f"cheap_batch: cannot read tasks: {exc}", file=sys.stderr)
        return 2
    if not tasks:
        print("{}")
        return 0

    facts = resolve(tasks, remote=args.remote, fmt="json" if args.json_out else None)

    if args.stats:
        served = {}
        for f in facts.values():
            served[f["tier"] or "unserved"] = served.get(f["tier"] or "unserved", 0) + 1
        print(f"cheap_batch: {len(tasks)} task(s) -> {served}", file=sys.stderr)
        for k, f in facts.items():
            if not f["ok"]:
                print(f"  unserved {k}: {f.get('why') or 'no reason given'}", file=sys.stderr)

    print(json.dumps(facts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
