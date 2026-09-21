#!/usr/bin/env python3
"""Offline demo of the offload tool loop (Loci issue #376).

Default mode is deterministic and needs no GPU or network: a scripted "model" drives
fake tools that return 2-6 KB payloads, and the metrics table shows what a cloud-driven
loop would have paid versus what the cloud caller pays to read the offloaded result.

    python scripts/offload_demo.py

--live does not run anything itself: the real tools only exist inside the Loci server
process, so call the MCP tool `offload_tool_loop` (see docs/OPERATIONS.md).
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp"))
import offload_loop as ol  # noqa: E402

TASK = "Which investigations mention 10.0.0.7 and what are their statuses?"
LIVE_HELP = """\
--live: call the MCP tool from a client connected to the Loci server, e.g.

  offload_tool_loop(
      task="%s",
      allowed_tools=["investigation_list", "investigation_entity_lookup",
                     "investigation_load"],
      max_steps=8)

Warm the model first (scripts/gpu_warm.py) so the ~70s cold load does not eat the
elapsed budget. Record status, metrics and the head of the audit JSONL (audit.path).
""" % TASK


def _payload(kb: int, tag: str) -> str:
    return json.dumps({"tag": tag, "rows": [f"{tag} evidence line " * 6] * (kb * 1024 // 120)})


def _intent(tool: str, **args) -> str:
    return json.dumps({"action": "tool_call", "tool": tool, "args": args})


def demo() -> dict:
    steps = [
        _intent("investigation_list", limit=20),
        _intent("investigation_entity_lookup", entity="10.0.0.7", entity_type="ip"),
        _intent("investigation_load", investigation_id="inv-a", fidelity="brief"),
        _intent("investigation_load", investigation_id="inv-b", fidelity="brief"),
        json.dumps({"action": "final_answer", "content":
                    "inv-a (open) and inv-b (fixed) mention 10.0.0.7."}),
    ]
    tools = {
        "investigation_list": lambda **kw: _payload(6, "list"),
        "investigation_entity_lookup": lambda **kw: _payload(5, "lookup"),
        "investigation_load": lambda **kw: _payload(3, kw["investigation_id"]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        mem = Path(tmp) / "mem"
        for inv in ("inv-a", "inv-b"):
            (mem / inv).mkdir(parents=True)
            (mem / inv / "manifest.json").write_text("{}")
        queue = list(steps)
        env = ol.run_loop(
            TASK, policy=ol.make_policy(["investigation_list", "investigation_entity_lookup",
                                         "investigation_load"]),
            model_fn=lambda prompt, timeout: {"text": queue.pop(0), "ok": True, "model": "scripted"},
            tools=tools, memory_dir=mem, audit=ol.jsonl_sink(Path(tmp) / "audit", "demo"))
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true", help="print how to run against the real lane")
    if ap.parse_args().live:
        print(LIVE_HELP)
        return 0
    env = demo()
    m = env["metrics"]
    print(f"status={env['status']} reason={env['reason']} steps={m['steps']} "
          f"tool_calls={m['tool_calls']}")
    print("\nESTIMATE (bytes/4), not billed tokens")
    rows = [("tool bytes produced (raw)", m["tool_bytes_raw"]),
            ("est tokens, cloud-only loop (baseline)", m["est_cloud_baseline_tokens"]),
            ("est tokens returned to cloud caller", m["est_tokens_returned"]),
            ("est tokens saved", m["est_tokens_saved"]),
            ("savings ratio", m["savings_ratio"])]
    for label, val in rows:
        print(f"  {label:<42}{val:>10}")
    return 0 if env["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
