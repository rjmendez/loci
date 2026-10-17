#!/usr/bin/env python3
"""
Qdrant snapshot cron — creates a snapshot of every collection, keeps N generations.

Qdrant stores snapshots server-side under <data_dir>/snapshots/<collection_name>/.
This script triggers the REST API to create a snapshot, then deletes excess
older snapshots from the same collection, keeping only the N most recent.

Usage:
    python3 qdrant_snapshot.py [--keep N] [--dry-run] [--collections c1,c2]

Configuration (env vars):
    QDRANT_URL      — required
    QDRANT_API_KEY  — optional
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY", "")
DEFAULT_KEEP = 3


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if QDRANT_KEY:
        h["api-key"] = QDRANT_KEY
    return h


def _get(path: str) -> dict:
    req = urllib.request.Request(f"{QDRANT_URL}{path}", headers=_headers())
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _post(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{QDRANT_URL}{path}", data=data, headers=_headers(), method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def _delete(path: str) -> dict:
    req = urllib.request.Request(
        f"{QDRANT_URL}{path}", headers=_headers(), method="DELETE"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def list_collections() -> list[str]:
    return [c["name"] for c in _get("/collections")["result"]["collections"]]


def list_snapshots(collection: str) -> list[dict]:
    result = _get(f"/collections/{collection}/snapshots")["result"]
    # Sort ascending by creation_time so oldest is first
    return sorted(result, key=lambda s: s.get("creation_time", ""))


def create_snapshot(collection: str, dry_run: bool) -> str | None:
    if dry_run:
        print(f"  [dry-run] would POST /collections/{collection}/snapshots")
        return None
    result = _post(f"/collections/{collection}/snapshots")["result"]
    return result.get("name", "?")


def delete_snapshot(collection: str, snapshot_name: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] would DELETE /collections/{collection}/snapshots/{snapshot_name}")
        return
    _delete(f"/collections/{collection}/snapshots/{snapshot_name}")


def snapshot_collection(collection: str, keep: int, dry_run: bool) -> None:
    print(f"\n[snapshot] {collection}")

    name = create_snapshot(collection, dry_run)
    if name:
        print(f"  created: {name}")

    snapshots = list_snapshots(collection)
    print(f"  total snapshots after creation: {len(snapshots) + (1 if dry_run else 0)}")

    # After creation there are len(snapshots)+1 total; prune the oldest
    to_delete = snapshots[: max(0, len(snapshots) - keep + 1)]
    for s in to_delete:
        snap_name = s["name"]
        print(f"  deleting (excess): {snap_name}")
        delete_snapshot(collection, snap_name, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qdrant collection snapshot cron")
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP,
                        help=f"Snapshots to retain per collection (default {DEFAULT_KEEP})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without executing them")
    parser.add_argument("--collections",
                        help="Comma-separated collection names (default: all)")
    args = parser.parse_args()

    if not QDRANT_URL:
        print("ERROR: QDRANT_URL is not set", file=sys.stderr)
        sys.exit(1)

    ts = datetime.now(timezone.utc).isoformat()
    print(f"[qdrant_snapshot] {ts}  keep={args.keep}  dry_run={args.dry_run}")

    try:
        if args.collections:
            collections = [c.strip() for c in args.collections.split(",") if c.strip()]
        else:
            collections = list_collections()
    except Exception as exc:
        print(f"ERROR: could not list collections: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[qdrant_snapshot] collections: {collections}")

    errors = []
    for col in collections:
        try:
            snapshot_collection(col, args.keep, args.dry_run)
        except Exception as exc:
            print(f"  ERROR on {col}: {exc}", file=sys.stderr)
            errors.append(col)

    if errors:
        print(f"\n[qdrant_snapshot] DONE with errors on: {errors}", file=sys.stderr)
        sys.exit(1)
    print(f"\n[qdrant_snapshot] DONE")


if __name__ == "__main__":
    main()
