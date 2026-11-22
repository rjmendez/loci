#!/usr/bin/env python3
"""Qdrant backup: snapshot each collection, copy the snapshot off the Qdrant volume, verify the copy.

Why not just scripts/qdrant_snapshot.py: that keeps snapshots server-side, on the same volume as the
data (the k3s PVC inside the WSL disk), so it does not survive losing that disk. This job also downloads
each snapshot to DEST (F:), checks size and sha256, and keeps a few generations there.

Scope: collections up to --max-points (default 150000). The multi-million-point collections (tens of GB
each) are skipped and named in the log and health file; they are not covered by this backup.

    python qdrant_backup.py                      # back up
    python qdrant_backup.py --dry-run
    python qdrant_backup.py --restore-test NAME  # restore NAME's newest copy into a throwaway collection, compare, delete it

Config: ~/.loci/backends.toml [qdrant] url / api_key (never printed). State: qdrant_backup_health.json,
qdrant_backup.log in ~/.loci. Exit 1 if any collection failed or the disk guard tripped.
"""
import argparse, datetime as dt, hashlib, json, os, shutil, sys, tomllib
import requests

HOME = os.path.expanduser("~")
LOCI = os.path.join(HOME, ".loci")
DEST = os.environ.get("QDRANT_BACKUP_DEST", r"F:\loci-data\qdrant-snapshots")
HEALTH = os.path.join(LOCI, "qdrant_backup_health.json")
LOG = os.path.join(LOCI, "qdrant_backup.log")
KEEP_LOCAL, KEEP_SERVER, MIN_FREE_GB = 3, 1, 25


def say(msg):
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def cfg():
    q = tomllib.load(open(os.path.join(LOCI, "backends.toml"), "rb"))["qdrant"]
    return q["url"].rstrip("/"), ({"api-key": q["api_key"]} if q.get("api_key") else {})


URL, HDR = cfg()


def api(method, path, **kw):
    kw.setdefault("timeout", 60)
    r = requests.request(method, URL + path, headers=HDR, **kw)
    r.raise_for_status()
    return r


def collections():
    out = []
    for c in api("GET", "/collections").json()["result"]["collections"]:
        n = api("GET", f"/collections/{c['name']}").json()["result"].get("points_count") or 0
        out.append((c["name"], n))
    return sorted(out)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def drop_server_snapshot(name, sname):
    """Best-effort delete of a snapshot THIS run created on the Qdrant host; never masks the error that got us here."""
    try:
        api("DELETE", f"/collections/{name}/snapshots/{sname}", timeout=120)
        say(f"  {name}: removed the unused server snapshot {sname}")
    except Exception as exc:
        say(f"  {name}: could not remove server snapshot {sname}: {type(exc).__name__}")


def backup_one(name, keep_local, dry):
    snaps = sorted(api("GET", f"/collections/{name}/snapshots").json()["result"], key=lambda s: s.get("creation_time", ""))
    if dry:
        say(f"  [dry-run] would snapshot {name} ({len(snaps)} on server), download and prune")
        return 0
    def count():
        return api("GET", f"/collections/{name}").json()["result"].get("points_count") or 0
    before = count()
    snap = api("POST", f"/collections/{name}/snapshots", timeout=900).json()["result"]
    after = count()
    sname, size = snap["name"], snap.get("size") or 0
    try:
        folder = os.path.join(DEST, name)
        os.makedirs(folder, exist_ok=True)
        free = shutil.disk_usage(DEST).free
        if free < size * 1.2 + MIN_FREE_GB * 2**30:
            raise RuntimeError(f"disk guard: {free / 2**30:.1f} GB free at {DEST}, snapshot is {size / 2**20:.0f} MB")
        final = os.path.join(folder, sname)
        part = final + ".part"
        with api("GET", f"/collections/{name}/snapshots/{sname}", stream=True, timeout=900) as r, open(part, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        got = os.path.getsize(part)
        if size and got != size:
            os.remove(part)
            raise RuntimeError(f"size mismatch: server {size}, downloaded {got}")
        digest = sha256(part)
        os.replace(part, final)
        with open(final + ".sha256", "w") as fh:
            fh.write(f"{digest}  {sname}\n")
    except BaseException:
        # the snapshot was created on the Qdrant host before the guard/download/verify; a failure here used to leave it there
        # (one per collection per failed night: 65 snapshots, 7.3 GiB inside the WSL disk by 2026-10-19)
        drop_server_snapshot(name, sname)
        raise
    # prune: the F: copy is the backup, so the server keeps only the newest KEEP_SERVER
    server = sorted(api("GET", f"/collections/{name}/snapshots").json()["result"], key=lambda s: s.get("creation_time", ""))
    for old in server[: max(0, len(server) - KEEP_SERVER)]:
        api("DELETE", f"/collections/{name}/snapshots/{old['name']}", timeout=120)
    local = sorted(f for f in os.listdir(folder) if f.endswith(".snapshot"))
    for old in local[: max(0, len(local) - keep_local)]:
        os.remove(os.path.join(folder, old))
        for ext in (".sha256", ".meta.json"):
            try:
                os.remove(os.path.join(folder, old + ext))
            except OSError:
                pass
    say(f"  {name}: {sname} {got / 2**20:.1f} MB sha256 {digest[:12]}")
    return got


def restore_test(name):
    folder = os.path.join(DEST, name)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".snapshot"))
    if not files:
        say(f"restore-test: no local snapshot for {name}")
        return 1
    path = os.path.join(folder, files[-1])
    want = open(path + ".sha256").read().split()[0]
    if sha256(path) != want:
        say("restore-test: sha256 mismatch on the stored copy")
        return 1
    tmp = f"zz_restore_test_{name}"[:60]
    try:
        meta = json.load(open(path + ".meta.json"))
        lo, hi = meta["points_before"], meta["points_after"]
    except (OSError, ValueError, KeyError):
        lo = hi = dict(collections()).get(name)
    try:
        with open(path, "rb") as fh:
            api("POST", f"/collections/{tmp}/snapshots/upload", params={"priority": "snapshot", "wait": "true"},
                files={"snapshot": (os.path.basename(path), fh)}, timeout=1800)
        n = api("GET", f"/collections/{tmp}").json()["result"].get("points_count")
        ok = lo <= n <= hi
        say(f"restore-test {name}: restored {n} points, {lo}..{hi} when the snapshot was taken: {'OK' if ok else 'MISMATCH'}")
        return 0 if ok else 1
    finally:
        try:
            api("DELETE", f"/collections/{tmp}", timeout=120)
            say(f"restore-test: throwaway collection {tmp} deleted")
        except Exception as exc:
            say(f"restore-test: could not delete {tmp}: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-points", type=int, default=150000)
    ap.add_argument("--keep", type=int, default=KEEP_LOCAL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore-test")
    ap.add_argument("--only", help="comma-separated collection names (default: all in scope)")
    a = ap.parse_args()
    if a.restore_test:
        return restore_test(a.restore_test)
    os.makedirs(DEST, exist_ok=True)
    cols = collections()
    only = {x.strip() for x in a.only.split(",")} if a.only else None
    chosen = [(n, p) for n, p in cols if p <= a.max_points and (only is None or n in only)]
    skipped = [{"name": n, "points": p} for n, p in cols if p > a.max_points]
    say(f"qdrant_backup: {len(chosen)} collections to back up, {len(skipped)} skipped as too large "
        f"(> {a.max_points} points): {[s['name'] for s in skipped]}")
    total, errors = 0, []
    for name, pts in chosen:
        try:
            total += backup_one(name, a.keep, a.dry_run)
        except Exception as exc:
            errors.append({"name": name, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
            say(f"  ERROR {name}: {errors[-1]['error']}")
    health = {"ok": not errors, "ts": dt.datetime.now().isoformat(timespec="seconds"), "backed_up": len(chosen) - len(errors),
              "bytes": total, "skipped_large": skipped, "errors": errors, "dest": DEST,
              "free_gb": round(shutil.disk_usage(DEST).free / 2**30, 1)}
    if not a.dry_run:
        json.dump(health, open(HEALTH, "w"), indent=1)
    say(f"qdrant_backup: {'OK' if not errors else 'FAILED'} {len(chosen) - len(errors)}/{len(chosen)} backed up, "
        f"{total / 2**20:.0f} MB, {health['free_gb']} GB free")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
