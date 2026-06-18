#!/usr/bin/env python3
"""
a2a_context_bridge.py — push recent Mnemosyne memories to all mesh peers via A2A.

Run as a cron job (every 15-30 min) to keep the mesh in sync.
Uses the Loci A2A server's context_broadcast skill so local storage
and peer fanout happen atomically server-side.

Env vars (from ~/.hermes/.env or ~/.hermes/profiles/{HERMES_PROFILE}/.env):
  HERMES_A2A_URL      Local A2A server endpoint (default: http://127.0.0.1:8201)
  HERMES_A2A_TOKEN    Bearer token for the local server
  BRIDGE_LOOKBACK_MIN How many minutes back to look for new memories (default: 30)
  BRIDGE_MIN_IMP      Minimum importance to bridge (default: 0.5)
  BRIDGE_MAX_ITEMS    Max memories to push per run (default: 20)
  MNEMOSYNE_DATA_DIR  Mnemosyne SQLite dir (default: ~/.hermes/mnemosyne/data)
  BRIDGE_STATE_FILE   Path to state file tracking last-synced timestamp
                      (default: ~/.hermes/bridge_state.json)
  PEER_A2A_URLS       Comma-separated peer endpoints (passed through to server)
  PEER_A2A_TOKEN      Shared peer token (passed through to server)

Usage:
  python3 a2a_context_bridge.py [--dry-run] [--verbose]
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
import sqlite3
import sys
import uuid
from pathlib import Path

try:
    import aiohttp
except ImportError:
    sys.exit("aiohttp required: pip install aiohttp")

# ── env load ─────────────────────────────────────────────────────────────────────
_profile = os.environ.get("HERMES_PROFILE", "")
_hermes_home = os.path.expanduser(os.environ.get("HERMES_HOME", "~/.hermes"))
_ENV = (
    os.path.join(_hermes_home, "profiles", _profile, ".env")
    if _profile
    else os.path.join(_hermes_home, ".env")
)
if os.path.exists(_ENV):
    for _l in open(_ENV):
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── config ───────────────────────────────────────────────────────────────────────
LOCAL_A2A_URL  = os.environ.get("HERMES_A2A_URL", "http://127.0.0.1:8201")
LOCAL_A2A_TOKEN = os.environ.get("HERMES_A2A_TOKEN", "changeme")
LOOKBACK_MIN   = int(os.environ.get("BRIDGE_LOOKBACK_MIN", "30"))
MIN_IMP        = float(os.environ.get("BRIDGE_MIN_IMP", "0.5"))
MAX_ITEMS      = int(os.environ.get("BRIDGE_MAX_ITEMS", "20"))
AGENT_ID       = os.environ.get("HERMES_AGENT_ID", "hermes")

_mnem_dir   = os.path.expanduser(os.environ.get("MNEMOSYNE_DATA_DIR", "~/.hermes/mnemosyne/data"))
MNEMOSYNE_DB= os.path.join(_mnem_dir, "mnemosyne.db")

_state_default = os.path.expanduser("~/.hermes/bridge_state.json")
STATE_FILE  = os.environ.get("BRIDGE_STATE_FILE", _state_default)

log = logging.getLogger("a2a_context_bridge")


# ── state helpers ─────────────────────────────────────────────────────────────────
def _load_state() -> dict:
    try:
        return json.loads(Path(STATE_FILE).read_text())
    except Exception:
        return {}


def _save_state(state: dict):
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(STATE_FILE).write_text(json.dumps(state, indent=2))


# ── Mnemosyne query ───────────────────────────────────────────────────────────────
def _fetch_recent_memories(since: str, min_importance: float, max_items: int) -> list[dict]:
    """
    Fetch working_memory rows newer than `since` (ISO timestamp) with importance >= min_importance.
    Falls back to memories table if working_memory doesn't exist.
    """
    if not os.path.exists(MNEMOSYNE_DB):
        log.warning("Mnemosyne DB not found: %s", MNEMOSYNE_DB)
        return []

    conn = sqlite3.connect(MNEMOSYNE_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    rows = []
    try:
        # Try working_memory first (primary store)
        try:
            rows = conn.execute(
                "SELECT id, content, importance, created_at, source FROM working_memory "
                "WHERE created_at > ? AND importance >= ? "
                "ORDER BY created_at DESC LIMIT ?",
                (since, min_importance, max_items)
            ).fetchall()
        except sqlite3.OperationalError:
            # Fallback to memories table
            rows = conn.execute(
                "SELECT id, content, importance, created_at, source FROM memories "
                "WHERE created_at > ? AND importance >= ? "
                "ORDER BY created_at DESC LIMIT ?",
                (since, min_importance, max_items)
            ).fetchall()
    finally:
        conn.close()

    return [dict(r) for r in rows]


# ── A2A call ──────────────────────────────────────────────────────────────────────
async def _broadcast_memory(session: aiohttp.ClientSession, mem: dict, dry_run: bool) -> dict:
    if dry_run:
        return {"status": "dry_run", "id": mem["id"], "content_len": len(mem["content"])}

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tasks/send",
        "params": {
            "skill_id": "context_broadcast",
            "message":  mem["content"],
            "input": {
                "content":    mem["content"],
                "source":     mem.get("source") or "bridge",
                "importance": float(mem.get("importance") or 0.5),
                "bank":       "default",
            },
            "sender": AGENT_ID,
        },
    }
    headers = {
        "Authorization": f"Bearer {LOCAL_A2A_TOKEN}",
        "Content-Type":  "application/json",
    }
    try:
        async with session.post(
            f"{LOCAL_A2A_URL.rstrip('/')}/a2a",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as r:
            data = await r.json()
            if r.status == 200:
                out = data.get("result", {}).get("output", {})
                ok_peers = sum(
                    1 for p in out.get("broadcast", []) if p.get("status") == "ok"
                )
                return {"status": "ok", "id": mem["id"], "peers_ok": ok_peers}
            return {"status": f"http_{r.status}", "id": mem["id"]}
    except Exception as e:
        return {"status": "error", "id": mem["id"], "error": str(e)}


# ── main ──────────────────────────────────────────────────────────────────────────
async def run(dry_run: bool, verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [bridge] %(levelname)s %(message)s",
    )

    state = _load_state()
    last_run = state.get("last_run")

    if last_run:
        since = last_run
        log.info("Fetching memories since last run: %s", since)
    else:
        # First run: look back LOOKBACK_MIN minutes
        since = (
            datetime.datetime.utcnow() - datetime.timedelta(minutes=LOOKBACK_MIN)
        ).isoformat()
        log.info("First run — looking back %d min (since %s)", LOOKBACK_MIN, since)

    mems = _fetch_recent_memories(since, MIN_IMP, MAX_ITEMS)
    log.info("Found %d memories to bridge (imp>=%.1f)", len(mems), MIN_IMP)

    if not mems:
        log.info("Nothing to bridge.")
        if not dry_run:
            state["last_run"] = datetime.datetime.utcnow().isoformat()
            _save_state(state)
        return

    ok = fail = 0
    async with aiohttp.ClientSession() as session:
        for mem in mems:
            result = await _broadcast_memory(session, mem, dry_run)
            if result.get("status") in ("ok", "dry_run"):
                ok += 1
                if verbose:
                    log.debug("  ok  id=%s peers=%s preview=%r",
                               result["id"], result.get("peers_ok", "?"),
                               mem["content"][:80])
            else:
                fail += 1
                log.warning("  FAIL id=%s status=%s err=%s",
                             result["id"], result.get("status"), result.get("error", ""))

    log.info("Bridge complete — ok=%d fail=%d dry_run=%s", ok, fail, dry_run)

    if not dry_run:
        state["last_run"] = datetime.datetime.utcnow().isoformat()
        state["last_ok"]  = ok
        state["last_fail"]= fail
        _save_state(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bridge Mnemosyne memories to A2A peers")
    parser.add_argument("--dry-run",  action="store_true", help="Print what would be sent, don't send")
    parser.add_argument("--verbose",  action="store_true", help="Debug logging")
    args = parser.parse_args()
    asyncio.run(run(dry_run=args.dry_run, verbose=args.verbose))
