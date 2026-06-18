#!/usr/bin/env python3
"""
qdrant_payload_indexes.py — one-shot idempotent setup for keyword payload indexes.

Creates keyword payload indexes on fields used for context-gated retrieval.
Index creation is a no-op if the index already exists, so this is safe to re-run.

Usage:
    python3 scripts/qdrant_payload_indexes.py

Env overrides:
    QDRANT_URL       — default http://localhost:6333
    QDRANT_API_KEY   — falls back to ~/.claude/settings.json mcpServers entry
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error

# ── config ────────────────────────────────────────────────────────────────────
_HOME = os.path.expanduser("~")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY", "")

if not QDRANT_KEY:
    try:
        import json as _json
        _cfg = _json.load(open(os.path.join(_HOME, ".claude", "settings.json")))
        QDRANT_KEY = _cfg["mcpServers"]["hermes_memory"]["env"]["QDRANT_API_KEY"]
    except Exception:
        pass

# ── index definitions ─────────────────────────────────────────────────────────
# (collection, field_name)
INDEXES = [
    ("mnemosyne",       "project"),
    ("mnemosyne",       "session_id"),
    ("hermes_sessions", "project"),
    ("hermes_sessions", "session_id"),
    ("hermes_sessions", "cwd"),
    ("hermes_memory",   "project"),
]


def create_index(collection: str, field_name: str) -> None:
    """PUT /collections/{collection}/index with keyword schema."""
    url = f"{QDRANT_URL.rstrip('/')}/collections/{collection}/index"
    payload = json.dumps({
        "field_name": field_name,
        "field_schema": "keyword",
    }).encode()

    headers = {"Content-Type": "application/json"}
    if QDRANT_KEY:
        headers["api-key"] = QDRANT_KEY

    req = urllib.request.Request(url, data=payload, headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            status = body.get("result", {}).get("status", body.get("status", "?"))
            print(f"[index] {collection}.{field_name} -> {status}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"[index] WARNING: collection '{collection}' not found — skipping",
                  file=sys.stderr)
        else:
            body = exc.read().decode(errors="replace")
            print(f"[index] ERROR {exc.code} on {collection}.{field_name}: {body}",
                  file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        print(f"[index] ERROR on {collection}.{field_name}: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    print(f"[index] Qdrant: {QDRANT_URL}")
    print(f"[index] Auth:   {'yes' if QDRANT_KEY else 'no (no api-key)'}")
    print()
    for collection, field_name in INDEXES:
        create_index(collection, field_name)
    print()
    print("[index] Done.")


if __name__ == "__main__":
    main()
