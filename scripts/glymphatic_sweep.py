#!/usr/bin/env python3
"""
Glymphatic sweep — offline memory maintenance and waste clearance.

Biological analog: the glymphatic system flushes metabolic waste (amyloid-β,
tau, lactate) during sleep via astrocyte-gated CSF flow through expanded
interstitial space. Clearance is ~2× more active during sleep and mutually
exclusive with active wakefulness.

This script runs in the offline window (cron, after SWR replay) and cleans:

  1. Superseded verdicts — for each subject_signature, keep the highest-
     confidence verdict; prune older/lower-confidence superseded records.
  2. Orphaned investigation sessions — sessions with no recall activity and
     no outgoing graph edges, older than ORPHAN_TTL_DAYS.
  3. Dangling graph edges — edges whose source/target node no longer exists
     in the mnemosyne SQLite DB.
  4. Near-duplicate Qdrant points — cosine~1.0 pairs: winner-take-all keeps
     the higher-importance point, prunes the redundant one.

Run from cron (low frequency, e.g. daily). Never run concurrently with SWR
replay or amem_consolidation — use the mutex flag to prevent races.

Usage:
    python3 glymphatic_sweep.py [--dry-run] [--skip STEP,STEP]
    Steps: verdicts, orphans, edges, duplicates
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

# ── config ────────────────────────────────────────────────────────────────────

QDRANT_URL   = os.environ.get("QDRANT_URL")
QDRANT_KEY   = os.environ.get("QDRANT_API_KEY",  "")
VERDICTS_COL = os.environ.get("VERDICTS_COL",    "hermes_verdicts")
MEMORY_COL   = os.environ.get("MEMORY_COL",      "hermes_memory")
DB_PATH      = os.environ.get("MNEMOSYNE_DB",
    os.path.expanduser("~/.hermes/mnemosyne/data/mnemosyne.db"))
MUTEX_FLAG   = os.environ.get("GLYMPHATIC_MUTEX",
    os.path.expanduser("~/.hermes/glymphatic.lock"))

ORPHAN_TTL_DAYS         = float(os.environ.get("GLYMPHATIC_ORPHAN_TTL_DAYS",   "7"))
DUPLICATE_COS_THRESHOLD = float(os.environ.get("GLYMPHATIC_DUP_COS_THRESH",    "0.98"))
SCROLL_BATCH            = int(os.environ.get("GLYMPHATIC_SCROLL_BATCH",         "500"))

_LN2 = math.log(2)


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def _hdrs() -> dict:
    h = {"Content-Type": "application/json"}
    if QDRANT_KEY:
        h["api-key"] = QDRANT_KEY
    return h


def _scroll_all(collection: str, with_vectors: bool = False) -> list[dict]:
    """Scroll all points in a collection. Returns list of {id, payload, vector?}."""
    if not QDRANT_URL:
        print("[glym] QDRANT_URL is not set — skipping Qdrant operations")
        return []
    url  = f"{QDRANT_URL}/collections/{collection}/points/scroll"
    pts  = []
    offset = None
    while True:
        body: dict = {"limit": SCROLL_BATCH, "with_payload": True, "with_vector": with_vectors}
        if offset is not None:
            body["offset"] = offset
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=_hdrs(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                d = json.loads(resp.read())
        except Exception as e:
            print(f"[glym] scroll {collection} failed: {e}", file=sys.stderr)
            break
        result = d.get("result") or {}
        pts.extend(result.get("points", []))
        offset = result.get("next_page_offset")
        if not offset:
            break
    return pts


def _delete_points(collection: str, ids: list, dry_run: bool) -> int:
    if not ids:
        return 0
    if not QDRANT_URL:
        return 0
    if dry_run:
        print(f"[glym] DRY-RUN would delete {len(ids)} pts from {collection}")
        return len(ids)
    url  = f"{QDRANT_URL}/collections/{collection}/points/delete"
    body = json.dumps({"points": ids}).encode()
    req  = urllib.request.Request(url, data=body, headers=_hdrs(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return len(ids)
    except Exception as e:
        print(f"[glym] delete failed: {e}", file=sys.stderr)
        return 0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


# ── step 1: superseded verdicts ───────────────────────────────────────────────

def sweep_verdicts(dry_run: bool) -> int:
    """Per signature: keep highest-confidence verdict, prune the rest."""
    print("[glym/verdicts] scanning …")
    pts = _scroll_all(VERDICTS_COL)
    if not pts:
        print("[glym/verdicts] collection empty or unreachable")
        return 0

    # Group by subject_signature.
    by_sig: dict[str, list[dict]] = {}
    for pt in pts:
        pl  = pt.get("payload") or {}
        sig = pl.get("subject_signature") or pl.get("id") or str(pt.get("id"))
        by_sig.setdefault(sig, []).append(pt)

    to_delete = []
    for sig, group in by_sig.items():
        if len(group) <= 1:
            continue
        # Keep the one with highest confidence; break ties by latest last_seen.
        def _rank(pt):
            pl = pt.get("payload") or {}
            conf = float(pl.get("confidence", 0.0) or 0.0)
            ts   = pl.get("last_seen") or pl.get("first_seen") or ""
            return (conf, ts)
        group.sort(key=_rank, reverse=True)
        losers = group[1:]
        to_delete.extend(p["id"] for p in losers)

    deleted = _delete_points(VERDICTS_COL, to_delete, dry_run)
    print(f"[glym/verdicts] {deleted} superseded verdicts removed "
          f"({len(by_sig)} signatures, {len(pts)} total)")
    return deleted


# ── step 2: orphaned sessions ─────────────────────────────────────────────────

def sweep_orphans(dry_run: bool) -> int:
    """Remove investigation session JSONL dirs with no recall and no edges."""
    if not os.path.exists(DB_PATH):
        print(f"[glym/orphans] DB not found: {DB_PATH}")
        return 0

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    try:
        cur.execute("SELECT DISTINCT source FROM graph_edges")
        has_edges = {str(row["source"]) for row in cur.fetchall()}
    except sqlite3.OperationalError:
        has_edges = set()

    cutoff_ts = time.time() - ORPHAN_TTL_DAYS * 86400

    try:
        cur.execute(
            "SELECT id, created_at FROM working_memory WHERE recall_count = 0 OR recall_count IS NULL"
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []

    orphan_ids = []
    for row in rows:
        rid = str(row["id"])
        if rid in has_edges:
            continue
        created = row["created_at"]
        try:
            ts = datetime.fromisoformat(str(created).replace("Z", "+00:00")).timestamp()
        except Exception:
            continue
        if ts < cutoff_ts:
            orphan_ids.append(rid)

    if orphan_ids and not dry_run:
        placeholders = ",".join("?" * len(orphan_ids))
        cur.execute(f"DELETE FROM working_memory WHERE id IN ({placeholders})", orphan_ids)
        conn.commit()
    conn.close()

    action = "DRY-RUN would remove" if dry_run else "removed"
    print(f"[glym/orphans] {action} {len(orphan_ids)} orphaned working_memory entries "
          f"(no recall, no edges, >{ORPHAN_TTL_DAYS}d old)")
    return len(orphan_ids)


# ── step 3: dangling graph edges ──────────────────────────────────────────────

def sweep_edges(dry_run: bool) -> int:
    """Remove graph_edges where source or target no longer exists in working_memory."""
    if not os.path.exists(DB_PATH):
        print(f"[glym/edges] DB not found: {DB_PATH}")
        return 0

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    try:
        cur.execute("SELECT DISTINCT id FROM working_memory")
        existing = {str(r["id"]) for r in cur.fetchall()}
    except sqlite3.OperationalError:
        conn.close()
        print("[glym/edges] working_memory table not found")
        return 0

    try:
        cur.execute("SELECT rowid, source, target FROM graph_edges")
        edges = cur.fetchall()
    except sqlite3.OperationalError:
        conn.close()
        print("[glym/edges] graph_edges table not found")
        return 0

    dangling_rowids = [
        row["rowid"] for row in edges
        if str(row["source"]) not in existing or str(row["target"]) not in existing
    ]

    if dangling_rowids and not dry_run:
        placeholders = ",".join("?" * len(dangling_rowids))
        cur.execute(f"DELETE FROM graph_edges WHERE rowid IN ({placeholders})", dangling_rowids)
        conn.commit()
    conn.close()

    action = "DRY-RUN would remove" if dry_run else "removed"
    print(f"[glym/edges] {action} {len(dangling_rowids)} dangling edges "
          f"(of {len(edges)} total)")
    return len(dangling_rowids)


# ── step 4: near-duplicate Qdrant points ─────────────────────────────────────

def sweep_duplicates(dry_run: bool) -> int:
    """WTA dedup: for cosine-near-1.0 pairs, keep higher importance point."""
    print("[glym/duplicates] loading hermes_memory with vectors …")
    pts = _scroll_all(MEMORY_COL, with_vectors=True)
    if not pts:
        print("[glym/duplicates] collection empty or unreachable")
        return 0

    # Filter to points that actually have a dense vector.
    def _vec(pt):
        v = pt.get("vector")
        if isinstance(v, dict):
            return v.get("dense") or v.get("") or next(iter(v.values()), None)
        return v if isinstance(v, list) else None

    vectored = [(pt, _vec(pt)) for pt in pts if _vec(pt)]
    print(f"[glym/duplicates] {len(vectored)} points with vectors")

    to_delete: set = set()
    n = len(vectored)
    for i in range(n):
        pt_a, vec_a = vectored[i]
        if pt_a["id"] in to_delete:
            continue
        pl_a = pt_a.get("payload") or {}
        imp_a = float(pl_a.get("importance", 0.5) or 0.5)
        for j in range(i + 1, n):
            pt_b, vec_b = vectored[j]
            if pt_b["id"] in to_delete:
                continue
            sim = _cosine(vec_a, vec_b)
            if sim >= DUPLICATE_COS_THRESHOLD:
                pl_b = pt_b.get("payload") or {}
                imp_b = float(pl_b.get("importance", 0.5) or 0.5)
                loser = pt_b["id"] if imp_a >= imp_b else pt_a["id"]
                to_delete.add(loser)

    deleted = _delete_points(MEMORY_COL, list(to_delete), dry_run)
    print(f"[glym/duplicates] {deleted} near-duplicate points removed "
          f"(cos >= {DUPLICATE_COS_THRESHOLD})")
    return deleted


# ── mutex ──────────────────────────────────────────────────────────────────────

class _Mutex:
    def __init__(self, path: str):
        self._path = path

    def __enter__(self):
        if os.path.exists(self._path):
            with open(self._path) as f:
                info = f.read().strip()
            # Parse PID from lock file contents (e.g. "pid=1234 ts=...")
            pid = None
            for part in info.split():
                if part.startswith("pid="):
                    try:
                        pid = int(part[4:])
                    except ValueError:
                        pass
                    break
            # Check if the owning process is still alive
            pid_alive = False
            if pid is not None:
                try:
                    import psutil
                    pid_alive = psutil.pid_exists(pid)
                except ImportError:
                    pid_alive = os.path.exists(f"/proc/{pid}")
            if pid_alive:
                raise RuntimeError(f"Another sweep is running (lock: {info}). Aborting.")
            # Stale lock — owning process is gone; remove it and continue
            try:
                os.remove(self._path)
            except OSError:
                pass
        with open(self._path, "w") as f:
            f.write(f"pid={os.getpid()} ts={int(time.time())}")
        return self

    def __exit__(self, *_):
        try:
            os.remove(self._path)
        except Exception:
            pass


# ── main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False, skip: set | None = None) -> None:
    skip = skip or set()
    t0 = time.time()
    print(f"[glym] starting  dry_run={dry_run}  skip={skip or 'none'}")

    with _Mutex(MUTEX_FLAG):
        totals = {}

        if "verdicts" not in skip:
            totals["verdicts"] = sweep_verdicts(dry_run)

        if "orphans" not in skip:
            totals["orphans"] = sweep_orphans(dry_run)

        if "edges" not in skip:
            totals["edges"] = sweep_edges(dry_run)

        if "duplicates" not in skip:
            totals["duplicates"] = sweep_duplicates(dry_run)

    elapsed = time.time() - t0
    print(f"[glym] done in {elapsed:.1f}s — {totals}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glymphatic memory maintenance sweep")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be removed without deleting")
    parser.add_argument("--skip", default="",
                        help="Comma-separated steps to skip: verdicts,orphans,edges,duplicates")
    args   = parser.parse_args()
    skip   = {s.strip() for s in args.skip.split(",") if s.strip()}
    main(dry_run=args.dry_run, skip=skip)
