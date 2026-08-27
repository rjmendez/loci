#!/usr/bin/env python3
"""
generate_memory_md.py
─────────────────────
Regenerates ~/loci_memory/memories/MEMORY.md from a PostgreSQL database
(optional, configure via LOCI_DB_* env vars) or, if unavailable,
from the local Mnemosyne SQLite DB + static identity block.

Usage:
    python3 ~/loci_memory/scripts/generate_memory_md.py
    python3 ~/loci_memory/scripts/generate_memory_md.py --dry-run   (print to stdout, no write)
    python3 ~/loci_memory/scripts/generate_memory_md.py --source sqlite  (force SQLite mode)
    python3 ~/loci_memory/scripts/generate_memory_md.py --source postgres  (force Postgres mode)

Output format: § separators, GROUP: VALUE lines (same as existing MEMORY.md)
"""

import argparse
import datetime
import os
import sqlite3
import sys

# ── paths ─────────────────────────────────────────────────────────────────────
OUTPUT_PATH = os.path.expanduser("~/loci_memory/memories/MEMORY.md")
SQLITE_DB   = os.path.expanduser(
    os.environ.get("MNEMOSYNE_DATA_DIR", "~/.hermes/mnemosyne/data")
).replace("~", os.path.expanduser("~")) + "/mnemosyne.db"

# ── PostgreSQL config (optional — set LOCI_DB_* vars in .env to enable) ──────
PG_HOST = os.environ.get("LOCI_DB_HOST", "localhost")
PG_PORT = int(os.environ.get("LOCI_DB_PORT", "5432"))
PG_USER = os.environ.get("LOCI_DB_USER", "")
PG_PASS = os.environ.get("LOCI_DB_PASS", "")
PG_DB   = os.environ.get("LOCI_DB_NAME", "")

# ── static identity block — customize for your deployment ────────────────────
# (group, value) tuples seeded into MEMORY.md; no credentials — reference env var names.
STATIC_IDENTITY: list[tuple[str, str]] = []

# ── static key facts (optional: seed known agents / infra) ───────────────────
# (group, value) tuples for known agents/infra; reference tokens by env var name, never inline.
STATIC_KEY_FACTS: list[tuple[str, str]] = []

# Optional static infra notes (no credentials).
STATIC_INFRA: list[tuple[None, str]] = []


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL source
# ─────────────────────────────────────────────────────────────────────────────

def _try_postgres():
    """
    Try to connect to the configured PostgreSQL database (LOCI_DB_* env vars).
    Returns a list of (group, value) tuples, or None if unavailable.
    """
    try:
        import psycopg2  # type: ignore
    except ImportError:
        return None

    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT,
            user=PG_USER, password=PG_PASS,
            dbname=PG_DB, connect_timeout=3
        )
        try:
            cur = conn.cursor()

            rows = []

            for table in ("user_profile", "memory_config", "agent_facts", "facts"):
                try:
                    cur.execute(f"SELECT * FROM {table} LIMIT 1;")
                    cols = [d[0] for d in cur.description]
                    cur.execute(f"SELECT * FROM {table};")
                    for row in cur.fetchall():
                        rec = dict(zip(cols, row))
                        group = rec.get("group") or rec.get("category") or rec.get("section") or "Key Facts"
                        value = rec.get("value") or rec.get("content") or str(rec)
                        rows.append((group, value))
                    break
                except Exception:
                    continue
        finally:
            conn.close()
        return rows if rows else None

    except Exception as e:
        print(f"[generate_memory_md] PostgreSQL unavailable: {e}", file=sys.stderr)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SQLite source — supplements static block with dynamic facts from mnemosyne.db
# ─────────────────────────────────────────────────────────────────────────────

def _load_sqlite_key_facts():
    """
    Read high-importance key facts from mnemosyne SQLite to supplement MEMORY.md.
    Returns a list of (group, value) tuples.
    """
    if not os.path.exists(SQLITE_DB):
        print(f"[generate_memory_md] SQLite DB not found: {SQLITE_DB}", file=sys.stderr)
        return []

    rows = []
    try:
        conn = sqlite3.connect(SQLITE_DB, timeout=5)
        conn.row_factory = sqlite3.Row

        try:
            mem_rows = conn.execute(
                "SELECT content, importance, created_at FROM memories "
                "WHERE importance >= 0.85 ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
            for r in mem_rows:
                rows.append(("Key Memories (high-importance)", r["content"][:200]))
        except Exception as e:
            print(f"[generate_memory_md] memories query: {e}", file=sys.stderr)

        try:
            triple_rows = conn.execute(
                "SELECT subject, predicate, object, valid_from FROM triples "
                "ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
            for r in triple_rows:
                rows.append((
                    "Knowledge Graph",
                    f"`{r['subject']}` —[{r['predicate']}]→ `{r['object']}` (from {r['valid_from']})"
                ))
        except Exception as e:
            print(f"[generate_memory_md] triples query: {e}", file=sys.stderr)

        conn.close()
    except Exception as e:
        print(f"[generate_memory_md] SQLite error: {e}", file=sys.stderr)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Render MEMORY.md
# ─────────────────────────────────────────────────────────────────────────────

def _render(all_rows: list[tuple]) -> str:
    """
    Render a list of (group, value) tuples into the MEMORY.md § format.
    group=None means a bare line (no prefix).
    """
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"<!-- AUTO-GENERATED by generate_memory_md.py — do not edit by hand -->",
        f"<!-- Last generated: {now} -->",
        f"<!-- To make permanent changes, write to the DB and re-run generate_memory_md.py -->",
    ]

    for group, value in all_rows:
        lines.append("§")
        if group:
            lines.append(f"{group}: {value}")
        else:
            lines.append(value)

    lines.append("§")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Regenerate MEMORY.md")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print to stdout instead of writing the file")
    parser.add_argument("--source", choices=["auto", "postgres", "sqlite"],
                        default="auto",
                        help="Data source: auto (try postgres, fall back to sqlite), postgres, sqlite")
    args = parser.parse_args()

    # ── gather rows ───────────────────────────────────────────────────────────
    dynamic_rows: list[tuple] = []

    if args.source in ("auto", "postgres"):
        pg_rows = _try_postgres()
        if pg_rows:
            print(f"[generate_memory_md] Using PostgreSQL ({len(pg_rows)} rows)", file=sys.stderr)
            dynamic_rows = pg_rows
        elif args.source == "postgres":
            print("[generate_memory_md] ERROR: PostgreSQL not available and --source=postgres forced",
                  file=sys.stderr)
            sys.exit(1)
        else:
            print("[generate_memory_md] PostgreSQL unavailable — falling back to SQLite", file=sys.stderr)

    if not dynamic_rows:
        sqlite_rows = _load_sqlite_key_facts()
        print(f"[generate_memory_md] SQLite mode: {len(sqlite_rows)} dynamic rows", file=sys.stderr)
        dynamic_rows = sqlite_rows

    all_rows: list[tuple] = []
    all_rows.extend(STATIC_IDENTITY)
    all_rows.extend(STATIC_KEY_FACTS)
    all_rows.extend(dynamic_rows)
    all_rows.extend(STATIC_INFRA)

    # ── render ────────────────────────────────────────────────────────────────
    content = _render(all_rows)

    if args.dry_run:
        print(content)
        return

    # ── write (idempotent) ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(content)

    print(f"[generate_memory_md] Written {len(content)} bytes to {OUTPUT_PATH}", file=sys.stderr)
    print(f"OK: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
