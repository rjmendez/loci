"""Command-line access to the live Loci MCP server (http://127.0.0.1:8000/mcp) for agents that have no
Loci MCP tools connected. Read-only helpers. The bearer token is read from ~/.claude.json and never printed.

    python loci_rag.py search "query" [--collections a,b] [--limit 6] [--chars 2500]
    python loci_rag.py tool <tool_name> '{"arg": "value"}'      # any Loci tool; use only read-only ones
    python loci_rag.py tools                                    # list tool names

`search` is rag_context_search in compact mode: cited one-liners from Qdrant (hybrid dense+sparse, reranked).
Default collection is loci_memory (findings, notes, session knowledge). Useful others:
  flybrain_code, flybrain_racing_code, dama_gotchi_code, dama_implementation_plans, bma360dx_code,
  hermes_sessions, copilot_claude_week_artifacts, mnemosyne, memgas_l1, memgas_l2, memgas_l3,
  ops_logs, ops_logs_agent, research_papers, ecc_skills, loci_findings, gl_decision_library (NOT searchable by text).
"""
import argparse
import json
import os
import re
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
URL = os.environ.get("LOCI_MCP_URL", "http://127.0.0.1:8000/mcp")


def _token() -> str:
    p = os.path.join(os.path.expanduser("~"), ".claude.json")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d["mcpServers"]["loci"]["headers"]["Authorization"].split(" ", 1)[1]


def _post(body, sid=None, token=None):
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
         "Authorization": "Bearer " + token}
    if sid:
        h["Mcp-Session-Id"] = sid
    r = urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers=h), timeout=120)
    txt = r.read().decode()
    m = re.search(r"data: (\{.*\})", txt)
    return r.headers.get("Mcp-Session-Id"), json.loads(m.group(1) if m else txt)


def call(method, params=None):
    tok = _token()
    sid, _ = _post({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                               "clientInfo": {"name": "loci-rag-cli", "version": "1"}}}, token=tok)
    return _post({"jsonrpc": "2.0", "id": 2, "method": method, "params": params or {}}, sid, tok)[1]


def tool(name, args):
    r = call("tools/call", {"name": name, "arguments": args})
    if "error" in r:
        return {"error": r["error"]}
    res = r["result"]["content"][0]["text"]
    try:
        d = json.loads(res)
        return json.loads(d["result"]) if isinstance(d.get("result"), str) else d
    except Exception:
        return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--collections", default="")
    s.add_argument("--limit", type=int, default=6)
    s.add_argument("--chars", type=int, default=2500)
    t = sub.add_parser("tool")
    t.add_argument("name")
    t.add_argument("args", nargs="?", default="{}")
    sub.add_parser("tools")
    a = ap.parse_args()
    if a.cmd == "tools":
        print("\n".join(sorted(x["name"] for x in call("tools/list")["result"]["tools"])))
    elif a.cmd == "tool":
        out = tool(a.name, json.loads(a.args))
        print(out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, indent=1)[:12000])
    else:
        args = {"query": a.query, "limit": a.limit, "mode": "compact", "budget_chars": a.chars}
        if a.collections:
            args["collections"] = [c.strip() for c in a.collections.split(",") if c.strip()]
        d = tool("rag_context_search", args)
        if isinstance(d, str):
            print(d[:6000])
            return
        print(d.get("context") or "(no results)")
        failed = d.get("collections_failed")
        if failed:
            print("collections_failed:", failed, d.get("collection_errors", [])[:2])
        print(f"[{d.get('result_count')} results; searched {d.get('collections_searched')}]")


if __name__ == "__main__":
    main()
