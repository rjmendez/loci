#!/usr/bin/env python3
"""Produce DETERMINISTIC code-graph facts for a workflow — zero tokens, no LLM.

The main loop runs this before a fan-out and passes the JSON as the workflow's
`args.graphFacts`. A task tagged `tier: 'graph'` in loci-native.js is then resolved
straight from these facts with NO agent spawned — replacing the "grep agent" for
mechanical inventory work with an exact query over the Kuzu code<->memory graph.

Spec (argv[1] or stdin) is a list of requests, each with a `key` and ONE of:
  {"key":"callsites","impact":"investigation_store"}   -> impact_report (callers/co-ref/findings)
  {"key":"module","subsystem":"mcp/server.py"}         -> subsystem_report (symbols/hotspots/boundaries)
  {"key":"dead","deadCode":true}                        -> dead_code_candidates

Emits: {"<key>": {"kind":..., "text":<human summary>, "raw":<full dict>}, ...}
Fail-open: a bad/empty request yields a text note, never aborts the batch.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))


def _find_graph() -> str | None:
    import glob
    for p in glob.glob(os.path.expanduser("~/.hermes/**/graph.kuzu"), recursive=True):
        return p
    return None


def _summarize_impact(sym: str, r: dict) -> str:
    resolved = [x.get("id") or x.get("name") for x in (r.get("resolved") or [])]
    if not resolved:
        return f"'{sym}': not found in the code graph."
    callers = r.get("direct_callers") or []
    # de-dup while preserving order, cap for readability
    seen, uniq = set(), []
    for c in callers:
        if c not in seen:
            seen.add(c); uniq.append(c)
    lines = [f"'{sym}' resolved to {len(resolved)} symbol(s): {', '.join(resolved[:4])}"]
    lines.append(f"direct callers ({len(uniq)}): {', '.join(uniq[:25])}"
                 + (" …" if len(uniq) > 25 else ""))
    if r.get("transitive_caller_count") is not None:
        lines.append(f"transitive callers: {r['transitive_caller_count']}")
    co = [c.get("name") for c in (r.get("co_referenced") or [])][:8]
    if co:
        lines.append(f"co-referenced symbols: {', '.join(x for x in co if x)}")
    if r.get("referencing_finding_count"):
        invs = [i.get("id") for i in (r.get("investigations") or [])]
        lines.append(f"referenced by {r['referencing_finding_count']} finding(s) "
                     f"in investigations: {', '.join(x for x in invs if x)[:200]}")
    return "\n".join(lines)


def _summarize_subsystem(path: str, r: dict) -> str:
    files = r.get("files") or []
    if not files:
        return f"'{path}': no files matched in the code graph."
    lines = [f"subsystem '{path}': {len(files)} file(s), {r.get('symbol_count')} symbols, "
             f"kinds={r.get('kinds')}"]
    hot = [(h.get("name"), h.get("findings")) for h in (r.get("hotspot_symbols") or [])][:8]
    if hot:
        lines.append("hotspots (symbol:findings): " + ", ".join(f"{n}:{c}" for n, c in hot))
    if r.get("inbound_callers"):
        lines.append(f"inbound callers (cross-boundary): {len(r['inbound_callers'])}")
    if r.get("outbound_callees"):
        lines.append(f"outbound callees (cross-boundary): {len(r['outbound_callees'])}")
    invs = [i.get("id") for i in (r.get("investigations") or [])]
    if invs:
        lines.append(f"investigations touching it: {', '.join(x for x in invs if x)[:200]}")
    return "\n".join(lines)


def _summarize_dead(r: dict) -> str:
    cands = r.get("candidates") or []
    names = [c.get("name") for c in cands][:40]
    return (f"dead-code candidates ({len(cands)}): " + ", ".join(x for x in names if x)
            + (" …" if len(cands) > 40 else "")) if cands else "no dead-code candidates."


def main() -> int:
    raw = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] not in ("-", "") else sys.stdin.read()
    try:
        spec = json.loads(raw)
    except Exception as exc:
        print(f"error: spec must be JSON ({exc})", file=sys.stderr)
        return 2
    if isinstance(spec, dict):
        spec = [spec]

    out: dict = {}
    db = _find_graph()
    if not db:
        for req in spec:
            out[req.get("key", "?")] = {"kind": "unavailable", "text": "code graph not found", "raw": {}}
        print(json.dumps(out, indent=2))
        print("[graph_facts: no graph.kuzu found]", file=sys.stderr)
        return 0

    from graph.kuzu_store import KuzuStore
    from graph import analytics as A
    ks = KuzuStore(db)

    for req in spec:
        key = req.get("key", "?")
        try:
            if "impact" in req:
                r = A.impact_report(ks, req["impact"])
                out[key] = {"kind": "impact", "text": _summarize_impact(req["impact"], r), "raw": r}
            elif "subsystem" in req:
                r = A.subsystem_report(ks, req["subsystem"])
                out[key] = {"kind": "subsystem", "text": _summarize_subsystem(req["subsystem"], r), "raw": r}
            elif req.get("deadCode"):
                r = A.dead_code_candidates(ks)
                out[key] = {"kind": "dead_code", "text": _summarize_dead(r), "raw": r}
            else:
                out[key] = {"kind": "noop", "text": "no recognized request field", "raw": {}}
        except Exception as exc:  # fail-open per request
            out[key] = {"kind": "error", "text": f"graph query failed: {exc}", "raw": {}}

    print(json.dumps(out, indent=2))
    print(f"[graph_facts: {len(out)} fact(s) from {db}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
