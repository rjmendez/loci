#!/usr/bin/env python3
"""
Mnemosyne activity watchdog — multi-bank.
Checks default bank (via mnemosyne CLI) and dama-gotchi bank (via SQLite).
Outputs a line only if working_memory has grown in any bank since last check.
Silent when idle — prevents agent cron from burning tokens.

A probe that could not run says so instead of reporting 0. 0 is a real
working_memory count, and it can never exceed the stored high-water mark, so a
broken probe reporting 0 makes this watchdog permanently silent — and silence is
the contract above for "nothing to do".
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
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def _write_state(path: str, count: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(str(count))


def get_default_wm_count() -> "tuple[int | None, str | None]":
    """working_memory count from the default bank via the mnemosyne CLI.

    Returns (count, None) or (None, reason). MNEM is an absolute path into
    another project's venv; check_output hides every symptom of it going stale.
    """
    try:
        out = subprocess.check_output(
            [MNEM, "stats"], stderr=subprocess.DEVNULL, text=True, timeout=8
        )
    except Exception as exc:
        return None, f"{MNEM} stats failed: {exc!r}"
    m = re.search(r"Working memory:\s*(\d+)", out)
    if not m:
        return None, f"{MNEM} stats printed no 'Working memory:' line"
    return int(m.group(1)), None


def get_damagotchi_wm_count() -> "tuple[int | None, str | None]":
    """working_memory count from the dama-gotchi bank via SQLite.

    A missing DB is 'the bank moved', not 'the bank is empty'.
    """
    if not os.path.exists(DAMA_DB):
        return None, f"bank DB not found: {DAMA_DB}"
    try:
        conn = sqlite3.connect(DAMA_DB)
        count = conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
        conn.close()
    except Exception as exc:
        return None, f"{DAMA_DB} query failed: {exc!r}"
    return int(count), None


# ── check both banks ──────────────────────────────────────────────────────────

BANKS = [
    ("default",    get_default_wm_count,    STATE_FILE_DEFAULT),
    ("dama-gotchi", get_damagotchi_wm_count, STATE_FILE_DAMAGOTCHI),
]

grew_parts = []
blind_parts = []

for bank_name, count_fn, state_file in BANKS:
    current, err = count_fn()
    if current is None:
        blind_parts.append(f"{bank_name} ({err})")
        continue
    last = _read_state(state_file)

    if current > last:
        _write_state(state_file, current)
        grew_parts.append(f"{bank_name}: {last}->{current} (+{current - last})")

lines = []
if grew_parts:
    detail = ", ".join(grew_parts)
    lines.append(f"working_memory grew: {detail}. Run mnemosyne_sleep now.")
if blind_parts:
    lines.append(
        "WARNING: working_memory could not be read for " + ", ".join(blind_parts)
        + ". This is NOT idle — that bank is unmonitored until the probe is fixed."
    )

if lines:
    print("\n".join(lines))
# else: silent — cron agent won't fire tokens
