"""Pull OPNsense Suricata alerts over SSH into Qdrant `suricata_events`.

Pull, not push: OPNsense keeps its own logs (eve.json rotates ~every 9 h, 4 kept), so a
missed run just catches up. The router only runs `tail | head | gzip` on an exact byte
range (OPNsense has no python); parsing and aggregation happen here.

Offsets are (inode, byte offset) per file, so rotation (eve.json -> eve.json.0) is followed
by inode, and a trailing partial line is never consumed. State is written only after the
Qdrant writes succeed; any ssh/Qdrant/Ollama failure exits non-zero and changes nothing.

Alerts aggregate per (signature, category, severity, app_proto, proto, iface) into the
existing suricata_events schema: counts and first/last seen are merged into the matching
point (payload update, no re-embed); new keys are embedded and inserted.

    python opnsense_pull.py --dry-run     # parse and report, write nothing
    python opnsense_pull.py               # real run
"""
import argparse
import collections
import datetime as dt
import gzip
import json
import os
import subprocess
import sys
import time
import uuid

import requests

HOME = os.path.expanduser("~")
QDRANT = os.environ.get("QDRANT_URL", "http://127.0.0.1:30633")
OLLAMAS = [os.environ.get("OLLAMA_URL", "http://127.0.0.1:11435"), "http://127.0.0.1:11440"]
EMBED_MODEL = "nomic-embed-text"
COLL = "suricata_events"
HOST = os.environ.get("OPNSENSE_SSH", "fw")
SSH = os.environ.get("SSH_EXE", r"C:\Windows\System32\OpenSSH\ssh.exe")
EVE = "/var/log/suricata/eve.json"
STATE = os.path.join(HOME, ".loci", "opnsense_pull_state.json")
LOCK = os.path.join(HOME, ".loci", "opnsense_pull.lock")
NS = uuid.uuid5(uuid.NAMESPACE_DNS, "opnsense-suricata-events.loci")
LOCK_STALE_S = 1800
SSH_TIMEOUT_S = 900


def log(msg, **kw):
    print(json.dumps({"t": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "msg": msg, **kw}), flush=True)


def ssh_run(script, binary=False, timeout=60):
    # OPNsense root's login shell is csh: hand the script to sh in single quotes (script has none).
    assert "'" not in script
    r = subprocess.run([SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST, f"sh -c '{script}'"],
                       capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ssh rc={r.returncode}: {r.stderr.decode(errors='replace')[:300]}")
    return r.stdout if binary else r.stdout.decode()


def list_segments():
    """eve.json files oldest -> newest as (name, inode, size)."""
    out = ssh_run(f'stat -f "%i %z %N" {EVE}.[0-9] {EVE} 2>/dev/null; true')
    segs = []
    for line in out.splitlines():
        ino, size, name = line.split(" ", 2)
        segs.append((name, int(ino), int(size)))
    # eve.json.N: higher N is older; eve.json is newest
    def age_rank(s):
        suffix = s[0].rsplit(".", 1)[-1]
        return -int(suffix) if suffix.isdigit() else 1
    return sorted(segs, key=age_rank)


def plan(segs, state):
    ino = state.get("inode")
    if ino is not None:
        for i, (name, inode, size) in enumerate(segs):
            if inode == ino:
                off = state.get("offset", 0)
                if size < off:
                    off = 0
                return [(s, off if j == 0 else 0) for j, s in enumerate(segs[i:])], False
        return [(s, 0) for s in segs], True    # remembered file rotated out of retention: gap
    return [(s, 0) for s in segs], False


def read_segment(name, off, size):
    """Yield (line_bytes) for complete lines in [off, size); the last item is the consumed byte count."""
    n = size - off
    p = subprocess.Popen([SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST,
                          f"sh -c 'tail -c +{off + 1} {name} | head -c {n} | gzip -1'"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    gz = gzip.GzipFile(fileobj=p.stdout)
    buf, consumed = b"", 0
    try:
        while True:
            blk = gz.read(1 << 20)
            if not blk:
                break
            buf += blk
            parts = buf.split(b"\n")
            buf = parts.pop()
            for line in parts:
                consumed += len(line) + 1
                yield line
    finally:
        err = p.stderr.read()
        p.stdout.close()
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"ssh read of {name} rc={rc}: {err.decode(errors='replace')[:300]}")
    yield consumed


def key_of(sig, cat, sev, app, proto, iface):
    return (sig, cat, sev, app, proto, iface)


def parse_ts(s):
    return int(dt.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp())


def aggregate(todo):
    agg = {}
    stats = collections.Counter()
    new_state = None
    for (name, inode, size), off in todo:
        consumed = 0
        for item in read_segment(name, off, size):
            if isinstance(item, int):
                consumed = item
                break
            stats["lines"] += 1
            if b'"event_type":"alert"' not in item:
                continue
            try:
                ev = json.loads(item)
                a = ev["alert"]
                k = key_of(a["signature"], a.get("category", ""), str(a.get("severity", "")),
                           ev.get("app_proto", ""), ev.get("proto", ""), ev.get("in_iface", ""))
                ts = parse_ts(ev["timestamp"])
            except (ValueError, KeyError):
                stats["bad_alert_lines"] += 1
                continue
            g = agg.setdefault(k, {"count": 0, "first": ts, "last": ts,
                                   "src": collections.Counter(), "dest": collections.Counter()})
            g["count"] += 1
            g["first"] = min(g["first"], ts)
            g["last"] = max(g["last"], ts)
            g["src"][ev.get("src_ip", "")] += 1
            g["dest"][f'{ev.get("dest_ip", "")}:{ev.get("dest_port", "")}'] += 1
            stats["alerts"] += 1
        new_state = {"inode": inode, "offset": off + consumed}
        stats["segments"] += 1
        stats["bytes"] += consumed
    return agg, stats, new_state


def qheaders():
    key = os.environ.get("QDRANT_API_KEY")
    if not key:
        import tomllib
        with open(os.path.join(HOME, ".loci", "backends.toml"), "rb") as f:
            key = tomllib.load(f)["qdrant"].get("api_key")
    return {"api-key": key} if key else {}


def embed(texts):
    last = None
    for base in OLLAMAS:
        try:
            r = requests.post(f"{base}/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=300)
            r.raise_for_status()
            vecs = r.json()["embeddings"]
            if len(vecs) != len(texts) or any(len(v) != 768 for v in vecs):
                raise RuntimeError(f"embedder at {base} returned wrong shape")
            return vecs
        except Exception as e:  # noqa: BLE001 - try the next endpoint
            last = e
    raise RuntimeError(f"embed failed on all endpoints: {last}")


def existing_points(h):
    out, off = {}, None
    while True:
        body = {"limit": 256, "with_payload": True, "with_vector": False, **({"offset": off} if off is not None else {})}
        r = requests.post(f"{QDRANT}/collections/{COLL}/points/scroll", headers=h, json=body, timeout=60)
        r.raise_for_status()
        res = r.json()["result"]
        for p in res["points"]:
            pl = p["payload"]
            if pl.get("alert_signature") and not pl.get("anomaly_event"):
                k = key_of(pl["alert_signature"], pl.get("alert_category", ""), str(pl.get("alert_severity", "")),
                           pl.get("app_proto", ""), pl.get("proto", ""), pl.get("iface", ""))
                out.setdefault(k, (p["id"], pl))
        off = res.get("next_page_offset")
        if off is None:
            return out


def text_of(k):
    sig, cat, sev, app, proto, iface = k
    return f"suricata alert signature={sig} category={cat} severity={sev} app={app} proto={proto} iface={iface}"


def top(counter):
    return [f"{v}x {k}" for k, v in counter.most_common(3)]


def write(agg, h):
    have = existing_points(h)
    updates, inserts = [], []
    for k, g in agg.items():
        extra = {"recent_top_src": top(g["src"]), "recent_top_dest": top(g["dest"]), "recent_window_end": g["last"]}
        if k in have:
            pid, pl = have[k]
            updates.append({"set_payload": {"points": [pid], "payload": {
                "count": int(pl.get("count", 0)) + g["count"],
                "first_seen": min(int(pl.get("first_seen") or g["first"]), g["first"]),
                "last_seen": max(int(pl.get("last_seen") or 0), g["last"]), **extra}}})
        else:
            inserts.append((k, g, extra))
    for i in range(0, len(updates), 100):
        r = requests.post(f"{QDRANT}/collections/{COLL}/points/batch?wait=true", headers=h,
                          json={"operations": updates[i:i + 100]}, timeout=120)
        r.raise_for_status()
    for i in range(0, len(inserts), 64):
        batch = inserts[i:i + 64]
        vecs = embed([text_of(k) for k, _, _ in batch])
        pts = []
        for (k, g, extra), v in zip(batch, vecs):
            sig, cat, sev, app, proto, iface = k
            pts.append({"id": str(uuid.uuid5(NS, "|".join(k))), "vector": v, "payload": {
                "text": text_of(k), "label": sig, "alert_signature": sig, "alert_category": cat,
                "alert_severity": sev, "anomaly_event": "", "anomaly_type": "", "app_proto": app,
                "proto": proto, "iface": iface, "dest": "", "first_seen": g["first"], "last_seen": g["last"],
                "count": g["count"], "operator_tag": None, **extra}})
        r = requests.put(f"{QDRANT}/collections/{COLL}/points?wait=true", headers=h, json={"points": pts}, timeout=120)
        r.raise_for_status()
    return len(updates), len(inserts)


def load_state():
    try:
        with open(STATE) as f:
            return json.load(f).get("suricata", {})
    except (OSError, ValueError):
        return {}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"suricata": st, "saved": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, f)
    os.replace(tmp, STATE)


def take_lock(path=LOCK):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        if time.time() - os.path.getmtime(path) > LOCK_STALE_S:
            os.remove(path)
    except OSError:
        pass
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        log("another run holds the lock; exiting")
        sys.exit(0)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="parse and report; write nothing, keep state")
    a = ap.parse_args()
    take_lock()
    try:
        state = load_state()
        segs = list_segments()
        if not segs:
            raise RuntimeError("no eve.json found on the router")
        todo, gap = plan(segs, state)
        if gap:
            log("WARNING: remembered file rotated out of retention; alerts in between are lost", state=state)
        log("plan", segments=[(s[0], o, s[2]) for s, o in todo], dry_run=a.dry_run)
        agg, stats, new_state = aggregate(todo)
        log("parsed", **stats, keys=len(agg))
        for k, g in sorted(agg.items(), key=lambda kv: -kv[1]["count"])[:8]:
            log("top", sig=k[0][:90], sev=k[2], count=g["count"])
        if a.dry_run:
            return
        if agg:
            upd, ins = write(agg, qheaders())
            log("written", updated=upd, inserted=ins)
        save_state(new_state)
        log("state saved", **new_state)
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        log("FAILED", error=str(e)[:500])
        sys.exit(1)
