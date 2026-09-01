#!/usr/bin/env python3
"""
a2a_context_bridge.py — push recent Mnemosyne memories to all mesh peers via A2A.

Run as a cron job (every 15-30 min) to keep the mesh in sync.
Uses the Loci A2A server's context_broadcast skill so local storage
and peer fanout happen atomically server-side.

Env vars (from ~/.hermes/.env or ~/.hermes/profiles/{HERMES_PROFILE}/.env):
  LOCI_A2A_URL      Local A2A server endpoint (default: http://127.0.0.1:8201)
  LOCI_A2A_TOKEN    Bearer token for the local server
                    (the legacy HERMES_* spelling is still accepted for both)
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
# The legacy spelling is read directly rather than through legacy_env.apply(): this
# script runs standalone from systemd and has no import path to mcp/. The unit this
# repo ships points EnvironmentFile= at a profile that supplies only HERMES_A2A_*,
# so reading the new name alone silently produced an empty token and no TOTP header.
LOCAL_A2A_URL  = (os.environ.get("LOCI_A2A_URL")
                  or os.environ.get("HERMES_A2A_URL")
                  or "http://127.0.0.1:8201")
# Empty, not "changeme": the server defaults the same variable to '' so an unset token fails closed.
LOCAL_A2A_TOKEN = (os.environ.get("LOCI_A2A_TOKEN")
                   or os.environ.get("HERMES_A2A_TOKEN") or "")
if not LOCAL_A2A_TOKEN:
    print('WARNING: LOCI_A2A_TOKEN is not set. The bridge will be rejected by any '
          'server that enforces bearer auth. Generate one with: '
          'python3 -c "import secrets;print(secrets.token_hex(32))"', flush=True)
LOOKBACK_MIN   = int(os.environ.get("BRIDGE_LOOKBACK_MIN", "30"))
MIN_IMP        = float(os.environ.get("BRIDGE_MIN_IMP", "0.5"))
MAX_ITEMS      = int(os.environ.get("BRIDGE_MAX_ITEMS", "20"))
AGENT_ID       = os.environ.get("HERMES_AGENT_ID", "hermes")

# The local server may enforce TOTP on /a2a, where the bearer alone 401s; empty seed sends no header.
LOCAL_A2A_TOTP_SEED = (os.environ.get("LOCI_A2A_TOTP_SEED")
                       or os.environ.get("HERMES_A2A_TOTP_SEED") or "").strip()

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
    """Same-directory temp + os.replace. This runs on a 10-minute timer, and a
    truncated state file reads back as {}: sent_ids empty re-broadcasts the whole
    lookback window to every peer, and a lost last_run drops the held watermark."""
    p = Path(STATE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / (p.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, p)


# ── Mnemosyne query ───────────────────────────────────────────────────────────────
# Re-bridging a memory already in flight through the mesh is what turns two nodes into a loop.
ECHO_SOURCE_PREFIXES = [
    p.strip() for p in os.environ.get(
        "BRIDGE_EXCLUDE_SOURCES", "bridge:,broadcast:,context_broadcast"
    ).split(",") if p.strip()
]


def _fetch_recent_memories(since: str, min_importance: float, max_items: int) -> list[dict]:
    """
    Fetch memories newer than `since` (ISO timestamp) with importance >= min_importance,
    excluding anything that arrived through the mesh (see ECHO_SOURCE_PREFIXES).

    Reads BOTH tiers. This used to read working_memory and fall back to `memories` only on
    OperationalError -- i.e. only if the table did not exist. working_memory does exist, and
    it is the small staging tier (2 rows on hugbot5000-jetson against 139 in `memories`), so
    the fallback was unreachable and the real corpus was never bridged at all.

    created_at is normalised before comparison. Mnemosyne writes mostly ISO-with-T but not
    exclusively (138 T-separated vs 1 space-separated on that same host), and a raw string
    compare puts every space-separated row below every T-separated one, because ' ' (0x20)
    sorts under 'T' (0x54). Those rows would be silently skipped forever once `since` is a
    T-format timestamp.
    """
    if not os.path.exists(MNEMOSYNE_DB):
        log.warning("Mnemosyne DB not found: %s", MNEMOSYNE_DB)
        return []

    echo_clause = ""
    params_tail: list = []
    if ECHO_SOURCE_PREFIXES:
        echo_clause = " AND source IS NOT NULL AND " + " AND ".join(
            ["source NOT LIKE ?"] * len(ECHO_SOURCE_PREFIXES)
        )
        # An exact name like context_broadcast needs no wildcard; a prefix like bridge: does.
        params_tail = [p + "%" if p.endswith(":") else p for p in ECHO_SOURCE_PREFIXES]

    conn = sqlite3.connect(MNEMOSYNE_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    out: list[dict] = []
    seen: set = set()
    since_norm = (since or "").replace(" ", "T")
    _ALLOWED_TABLES = {"memories", "working_memory"}
    try:
        for table in ("memories", "working_memory"):
            if table not in _ALLOWED_TABLES:
                log.warning("skipping unknown table %s", table)
                continue
            try:
                rows = conn.execute(
                    f"SELECT id, content, importance, created_at, source FROM {table} "
                    "WHERE REPLACE(created_at, ' ', 'T') > ? AND importance >= ?"
                    + echo_clause +
                    " ORDER BY created_at DESC LIMIT ?",
                    (since_norm, min_importance, *params_tail, max_items)
                ).fetchall()
            except sqlite3.OperationalError as e:
                log.debug("skipping %s: %s", table, e)
                continue
            for r in rows:
                d = dict(r)
                if d["id"] in seen:
                    continue
                seen.add(d["id"])
                out.append(d)
    finally:
        conn.close()

    out.sort(key=lambda m: (m.get("created_at") or "").replace(" ", "T"), reverse=True)
    return out[:max_items]


# ── A2A call ──────────────────────────────────────────────────────────────────────
def _totp_now() -> str:
    """
    Current TOTP code for the local A2A server, or "" when no seed is configured.

    Fails loud rather than silently sending an unauthenticated request: if a seed IS set
    but pyotp is missing, every send would 401 and the only symptom would be a fail count
    in the log, which is the failure mode this helper exists to remove.
    """
    if not LOCAL_A2A_TOTP_SEED:
        return ""
    try:
        import pyotp
    except ImportError:
        raise SystemExit(
            "LOCI_A2A_TOTP_SEED is set but pyotp is not installed — the local server "
            "enforces TOTP and every send would 401. Install pyotp or unset the seed."
        )
    return pyotp.TOTP(LOCAL_A2A_TOTP_SEED).now()


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
                # Unstamped, a bridged memory looks locally-authored at the peer and bounces back.
                "source":     f"bridge:{AGENT_ID}",
                "importance": float(mem.get("importance") or 0.5),
                "bank":       "default",
                # context_broadcast stores before it fans out, and we relay through our OWN server.
                "store_local": False,
            },
            "sender": AGENT_ID,
        },
    }
    headers = {
        "Authorization": f"Bearer {LOCAL_A2A_TOKEN}",
        "Content-Type":  "application/json",
    }
    # Per send, not per run: a run spanning a 30s TOTP step would start failing halfway through.
    totp_code = _totp_now()
    if totp_code:
        headers["X-TOTP"] = totp_code
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
        since = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=LOOKBACK_MIN)
        ).isoformat()
        log.info("First run — looking back %d min (since %s)", LOOKBACK_MIN, since)

    mems = _fetch_recent_memories(since, MIN_IMP, MAX_ITEMS)
    log.info("Found %d memories to bridge (imp>=%.1f)", len(mems), MIN_IMP)

    if not mems:
        log.info("Nothing to bridge.")
        if not dry_run:
            state["last_run"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            _save_state(state)
        return

    # Ids already delivered, so holding the watermark back to retry does not re-send.
    sent_ids = list(state.get("sent_ids") or [])
    sent_set = set(sent_ids)

    ok = fail = skipped = 0
    async with aiohttp.ClientSession() as session:
        for mem in mems:
            if mem["id"] in sent_set:
                skipped += 1
                continue
            result = await _broadcast_memory(session, mem, dry_run)
            if result.get("status") in ("ok", "dry_run"):
                ok += 1
                if not dry_run:
                    sent_set.add(mem["id"])
                    sent_ids.append(mem["id"])
                if verbose:
                    log.debug("  ok  id=%s peers=%s preview=%r",
                               result["id"], result.get("peers_ok", "?"),
                               mem["content"][:80])
            else:
                fail += 1
                log.warning("  FAIL id=%s status=%s err=%s",
                             result["id"], result.get("status"), result.get("error", ""))

    log.info("Bridge complete — ok=%d fail=%d skipped=%d dry_run=%s", ok, fail, skipped, dry_run)

    if not dry_run:
        # Clean runs only: advancing past a failed send drops those memories for good.
        if fail == 0:
            state["last_run"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        else:
            log.warning(
                "Holding watermark at %s — %d send(s) failed and would otherwise be "
                "skipped permanently. They retry next tick.", state.get("last_run", since), fail
            )
            state.setdefault("last_run", since)
        state["last_ok"]   = ok
        state["last_fail"] = fail
        state["sent_ids"]  = sent_ids[-1000:]
        _save_state(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bridge Mnemosyne memories to A2A peers")
    parser.add_argument("--dry-run",  action="store_true", help="Print what would be sent, don't send")
    parser.add_argument("--verbose",  action="store_true", help="Debug logging")
    args = parser.parse_args()
    asyncio.run(run(dry_run=args.dry_run, verbose=args.verbose))
