#!/usr/bin/env python3
"""Print one query + its candidate pool from a judge_eval --prep file, for a judge agent.
Usage: judge_show.py <prep_or_input.json> <query_id>"""
import json
import sys

d = json.load(open(sys.argv[1]))
q = next((x for x in d["queries"] if x["id"] == sys.argv[2]), None)
if not q:
    print("query not found", file=sys.stderr)
    raise SystemExit(1)
print("QUERY:\n" + q["query"])
print("\nCANDIDATE DOCUMENTS (id then text):")
for p in q.get("pool", []):
    print(f"\n[{p['id']}]\n{p['text']}")
