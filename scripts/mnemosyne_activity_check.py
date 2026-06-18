#!/usr/bin/env python3
"""
Mnemosyne activity watchdog — multi-bank.
Checks default bank (via mnemosyne CLI) and dama-gotchi bank (via SQLite).
Outputs a line only if working_memory has grown in any bank since last check.
Silent when idle — prevents agent cron from burning tokens.
"""
import os
import re
import sqlite3
import subprocess

MNEM = os.path.expanduser("~/.hermes/hermes-agent/venv/bin/mnemosyne")
STATE_DIR = os.path.expanduser("~/.hermes/mnemosyne")

STATE_FILE_DEFAULT   = os.path.join(STATE_DIR, "last_wm_count.txt")
STATE_FILE_DAMAGOTCHI = os.path.join(STATE_DIR, "last_wm_count_dama-gotchi.txt")
DAMA_DB = os.path.expanduser(
    "~/.hermes/mnemosyne/data/banks/dama-gotchi/mnemosyne.db"
)


def _read_state(path: str) -> int:
    try:
        return int(open(path).read().strip())
    except Exception:
        return 0


def _write_state(path: str, count: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(str(count))


def get_default_wm_count() -> int:
    """Get working_memory count from default bank via mnemosyne CLI."""
    try:
        out = subprocess.check_output(
            [MNEM, "stats"], stderr=subprocess.DEVNULL, text=True, timeout=8
        )
        m = re.search(r"Working memory:\s*(\d+)", out)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


def get_damagotchi_wm_count() -> int:
    """Get working_memory count from dama-gotchi bank via SQLite."""
    if not os.path.exists(DAMA_DB):
        return 0
    try:
        conn = sqlite3.connect(DAMA_DB)
        count = conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
        conn.close()
        return int(count)
    except Exception:
        return 0


# ── check both banks ──────────────────────────────────────────────────────────

BANKS = [
    ("default",    get_default_wm_count,    STATE_FILE_DEFAULT),
    ("dama-gotchi", get_damagotchi_wm_count, STATE_FILE_DAMAGOTCHI),
]

grew_parts = []

for bank_name, count_fn, state_file in BANKS:
    current = count_fn()
    last = _read_state(state_file)

    if current > last:
        _write_state(state_file, current)
        grew_parts.append(f"{bank_name}: {last}->{current} (+{current - last})")

if grew_parts:
    detail = ", ".join(grew_parts)
    print(f"working_memory grew: {detail}. Run mnemosyne_sleep now.")
# else: silent — cron agent won't fire tokens
