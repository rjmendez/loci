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
import os as _os
import sys as _sys

# mcp/ holds the shared vector helpers; these scripts run standalone.
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "mcp"))
import vecmath as _vecmath  # noqa: E402
import os
import fcntl
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

# ── config ────────────────────────────────────────────────────────────────────

QDRANT_URL   = os.environ.get("QDRANT_URL")
QDRANT_KEY   = os.environ.get("QDRANT_API_KEY",  "")
VERDICTS_COL = os.environ.get("VERDICTS_COL",    "loci_verdicts")
MEMORY_COL   = os.environ.get("MEMORY_COL",      "loci_memory")
DB_PATH      = os.environ.get("MNEMOSYNE_DB",
    os.path.expanduser("~/.hermes/mnemosyne/data/mnemosyne.db"))
MUTEX_FLAG   = os.environ.get("GLYMPHATIC_MUTEX",
    os.path.expanduser("~/.hermes/glymphatic.lock"))

ORPHAN_TTL_DAYS         = float(os.environ.get("GLYMPHATIC_ORPHAN_TTL_DAYS",   "7"))
DUPLICATE_COS_THRESHOLD = float(os.environ.get("GLYMPHATIC_DUP_COS_THRESH",    "0.98"))
SCROLL_BATCH            = int(os.environ.get("GLYMPHATIC_SCROLL_BATCH",         "500"))
SHIFT_SNAPSHOT_FILE     = os.environ.get("GLYMPHATIC_SHIFT_SNAPSHOT",
    os.path.expanduser("~/.hermes/glymphatic_centroid_snapshot.json"))
SHIFT_THRESHOLD         = float(os.environ.get("GLYMPHATIC_SHIFT_THRESHOLD",   "0.05"))  # centroid drift > 5% triggers

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


def _cosine(a: list[float], b: list[float]):
    """Delegates to mcp/vecmath.py. None when the comparison is unanswerable —
    most importantly on a length mismatch, which this used to answer with a
    plausible number computed over the shorter vector's prefix."""
    return _vecmath.cosine(a, b)


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
            dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            # SQLite CURRENT_TIMESTAMP is naive UTC; .timestamp() would otherwise
            # read it as local wall-clock and skew the TTL by the UTC offset.
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts = dt.timestamp()
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
    print("[glym/duplicates] loading loci_memory with vectors …")
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
            # This branch DELETES a memory, so an unanswerable comparison must
            # never reach the threshold. A 768/384 mismatch used to score 0.707
            # here, which clears DUPLICATE_COS_THRESHOLD.
            if sim is None:
                continue
            if sim >= DUPLICATE_COS_THRESHOLD:
                pl_b = pt_b.get("payload") or {}
                imp_b = float(pl_b.get("importance", 0.5) or 0.5)
                loser = pt_b["id"] if imp_a >= imp_b else pt_a["id"]
                to_delete.add(loser)

    deleted = _delete_points(MEMORY_COL, list(to_delete), dry_run)
    print(f"[glym/duplicates] {deleted} near-duplicate points removed "
          f"(cos >= {DUPLICATE_COS_THRESHOLD})")
    return deleted


# ── step 5: content-shift detection (GAM pattern) ────────────────────────────

def _embed_text(text: str, ollama_url: str) -> list[float] | None:
    """Embed a single text via Ollama. Returns None on failure."""
    try:
        payload = json.dumps({"model": "nomic-embed-text", "input": text[:2000]}).encode()
        req = urllib.request.Request(
            f"{ollama_url.rstrip('/')}/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
        return body["data"][0]["embedding"]
    except Exception:
        return None


def _cosine_vecs(a: list[float], b: list[float]):
    """Same contract as _cosine. The `or 1.0` this replaced turned a zero-magnitude
    vector into a unit one and returned 0.0 — a drift reading manufactured from a
    vector that carried nothing."""
    return _vecmath.cosine(a, b)


def check_content_shift(ollama_url: str | None = None, sample_n: int = 50) -> dict:
    """Compute centroid drift between current working_memory and last snapshot.

    Returns {"drift_score": float, "should_sweep": bool, "n_sampled": int}.
    If Ollama is unreachable, returns {"drift_score": None, "should_sweep": False}.

    On the first run (no snapshot), saves a baseline and returns should_sweep=False.
    """
    _ollama = ollama_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    if not os.path.exists(DB_PATH):
        return {"drift_score": None, "should_sweep": False, "reason": "db_not_found"}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT content FROM working_memory ORDER BY created_at DESC LIMIT ?",
            (sample_n,),
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return {"drift_score": None, "should_sweep": False, "reason": "table_not_found"}
    finally:
        conn.close()

    if len(rows) < 5:
        return {"drift_score": None, "should_sweep": False, "reason": "too_few_rows"}

    # Embed a sample and compute centroid
    vecs = []
    for row in rows[:sample_n]:
        v = _embed_text(row["content"], _ollama)
        if v:
            vecs.append(v)
    if not vecs:
        return {"drift_score": None, "should_sweep": False, "reason": "embed_failed"}

    dim = len(vecs[0])
    centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in centroid)) or 1.0
    centroid = [x / norm for x in centroid]

    # Load snapshot
    if os.path.exists(SHIFT_SNAPSHOT_FILE):
        with open(SHIFT_SNAPSHOT_FILE) as fh:
            snapshot = json.load(fh)
        prev_centroid = snapshot.get("centroid", [])
        if prev_centroid and len(prev_centroid) == dim:
            drift = 1.0 - _cosine_vecs(centroid, prev_centroid)
            should_sweep = drift >= SHIFT_THRESHOLD
            # Update snapshot
            with open(SHIFT_SNAPSHOT_FILE, "w") as fh:
                json.dump({"centroid": centroid, "n_sampled": len(vecs), "updated_at": time.time()}, fh)
            print(f"[glym/shift] centroid drift={drift:.4f} threshold={SHIFT_THRESHOLD} "
                  f"should_sweep={should_sweep}")
            return {"drift_score": drift, "should_sweep": should_sweep, "n_sampled": len(vecs)}

    # First run — save baseline
    with open(SHIFT_SNAPSHOT_FILE, "w") as fh:
        json.dump({"centroid": centroid, "n_sampled": len(vecs), "updated_at": time.time()}, fh)
    print(f"[glym/shift] baseline snapshot saved (n={len(vecs)})")
    return {"drift_score": 0.0, "should_sweep": False, "n_sampled": len(vecs), "reason": "baseline_saved"}


# ── mutex ──────────────────────────────────────────────────────────────────────

class _Mutex:
    """Exclusive lock via flock() on a fixed lock file.

    flock is atomic at the kernel level (no check-then-act window) and is
    released automatically when the holding process's fd closes -- including
    on crash or kill -- so a dead owner's lock is never "stale": the OS has
    already freed it. That removes the need for any PID-liveness bookkeeping.
    """

    def __init__(self, path: str):
        self._path = path
        self._fd = None

    def __enter__(self):
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                info = os.read(fd, 256).decode(errors="replace").strip()
            except OSError:
                info = "<unknown>"
            os.close(fd)
            raise RuntimeError(f"Another sweep is running (lock: {info}). Aborting.")
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} ts={int(time.time())}".encode())
        self._fd = fd
        return self

    def __exit__(self, *_):
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        try:
            os.remove(self._path)
        except OSError:
            pass
        self._fd = None


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    dry_run: bool = False,
    skip: set | None = None,
    check_shift: bool = False,
    shift_only: bool = False,
    ollama_url: str | None = None,
) -> None:
    skip = skip or set()
    t0 = time.time()

    if check_shift or shift_only:
        shift_result = check_content_shift(ollama_url=ollama_url)
        if shift_only:
            print(f"[glym] shift check: {shift_result}")
            return
        if not shift_result.get("should_sweep", True):
            # drift_score is None on every early return (db_not_found, table_not_found,
            # too_few_rows, embed_failed); report the reason those carry instead.
            drift = shift_result.get("drift_score")
            detail = f"{drift:.4f}" if drift is not None else shift_result.get("reason", "unknown")
            print(f"[glym] content shift below threshold ({detail}) — skipping sweep")
            return
        print(f"[glym] content shift triggered sweep (drift={shift_result.get('drift_score'):.4f})")

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
    parser.add_argument("--check-shift", action="store_true",
                        help="Run content-shift detection before sweep; skip if below threshold")
    parser.add_argument("--shift-only", action="store_true",
                        help="Only run content-shift detection and exit (no sweep)")
    parser.add_argument("--ollama", default=None,
                        help="Ollama base URL for content-shift embeddings")
    args   = parser.parse_args()
    skip   = {s.strip() for s in args.skip.split(",") if s.strip()}
    main(
        dry_run=args.dry_run,
        skip=skip,
        check_shift=args.check_shift,
        shift_only=args.shift_only,
        ollama_url=args.ollama,
    )
