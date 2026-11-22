"""Pull OPNsense pf filter-log DAILY SUMMARIES over SSH into Qdrant `opnsense_filter`.

Raw filter lines (~1.3 M / 350 MB a day) are never stored: each (UTC day, interface) becomes one
summary point -- totals by action/direction/proto, top blocked sources, top blocked destination
ports, top rules (with the rule's own description), top internal talkers. Shares the ssh byte-range
reader with opnsense_pull.py (router has no python: it only runs tail | head | gzip).

State holds the open day's cumulative counters plus (file, byte offset). Every run re-writes the
touched days' points under deterministic ids from the CUMULATIVE counters, so a retry or a crash
between the write and the state save can never double count. Past days are finalised and dropped
from state when a newer file exists. Failure anywhere exits non-zero and leaves state untouched.

    python opnsense_filter_pull.py --dry-run --since 20261017
    python opnsense_filter_pull.py                  # backfill everything retained, then incremental
"""
import argparse
import collections
import datetime as dt
import json
import os
import re
import sys
import uuid

import requests

import opnsense_pull as base

HOME = base.HOME
COLL = "opnsense_filter"
STATE = os.path.join(HOME, ".loci", "opnsense_filter_state.json")
LOCK = os.path.join(HOME, ".loci", "opnsense_filter_pull.lock")
NS = uuid.uuid5(uuid.NAMESPACE_DNS, "opnsense-filter-daily.loci")
LOGDIR = "/var/log/filter"
TOPN = 25
IFACE_NAMES = {"ix1": "wan", "mlxen0": "lan", "lo0": "loopback", "wg0": "phone-wg"}
log = base.log

LABEL_RX = re.compile(r'label "([0-9a-f]{32})"\s*(?:#\s*(.*))?$')


def rule_labels():
    out = base.ssh_run('grep -o -E "label \\"[0-9a-f]{32}\\".*" /tmp/rules.debug 2>/dev/null; true')
    m = {}
    for line in out.splitlines():
        r = LABEL_RX.search(line)
        if r:
            m[r.group(1)] = (r.group(2) or "").strip()[:80]
    return m


def list_files():
    out = base.ssh_run(f'stat -f "%z %N" {LOGDIR}/filter_*.log 2>/dev/null; true')
    files = []
    for line in out.splitlines():
        size, name = line.split(" ", 1)
        files.append((name.rsplit("/", 1)[-1], int(size)))
    return sorted(files)


def new_bucket():
    return {"tot": collections.Counter(), "block_src": collections.Counter(), "block_dport": collections.Counter(),
            "block_rule": collections.Counter(), "pass_src": collections.Counter(), "pass_dport": collections.Counter(),
            "lines": 0}


def bucket_to_json(b):
    return {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in b.items()}


def bucket_from_json(d):
    b = new_bucket()
    for k, v in d.items():
        b[k] = collections.Counter(v) if isinstance(v, dict) else v
    return b


def feed(agg, line):
    """Fold one pf filterlog line into agg[(day, iface)]. Returns False if unparseable."""
    try:
        head, _, rest = line.partition(b"] ")
        day = head.split(b" ", 2)[1][:10].decode()
        f = rest.decode("ascii", "replace").split(",")
        iface, action, direction, ipver = f[4], f[6], f[7], f[8]
        if ipver == "4":
            proto, src, dst = f[16], f[18], f[19]
            sport, dport = (f[20], f[21]) if len(f) > 21 and proto in ("tcp", "udp") else ("", "")
        else:
            proto, src, dst = f[12], f[15], f[16]
            sport, dport = (f[17], f[18]) if len(f) > 18 and proto in ("tcp", "udp") else ("", "")
        tracker = f[3]
    except (IndexError, ValueError):
        return False
    b = agg.get((day, iface))
    if b is None:
        b = agg[(day, iface)] = new_bucket()
    b["lines"] += 1
    b["tot"][f"{action}|{direction}|v{ipver}|{proto}"] += 1
    pd = f"{proto}/{dport}" if dport else proto
    if action == "block":
        b["block_rule"][tracker] += 1
        if direction == "in":
            b["block_src"][src] += 1
            b["block_dport"][pd] += 1
    else:
        if direction == "in":
            b["pass_src"][src] += 1
        b["pass_dport"][pd] += 1
    return True


def plan(files, state):
    cur = state.get("file")
    out = []
    for name, size in files:
        if cur and name < cur:
            continue
        off = state.get("offset", 0) if name == cur else 0
        if name == cur and size < off:
            off = 0
        if size > off:
            out.append((name, off, size))
    return out


def topn(counter, labels=None):
    rows = []
    for k, v in counter.most_common(TOPN):
        rows.append([k, v, labels.get(k, "")] if labels is not None else [k, v])
    return rows


def render(day, iface, b, labels, partial):
    tot = b["tot"]
    n_block = sum(v for k, v in tot.items() if k.startswith("block|"))
    n_pass = sum(v for k, v in tot.items() if k.startswith("pass|"))
    name = IFACE_NAMES.get(iface, iface)
    bs = ", ".join(f"{k} x{v}" for k, v in b["block_src"].most_common(8))
    bp = ", ".join(f"{k} x{v}" for k, v in b["block_dport"].most_common(8))
    br = "; ".join(f'{labels.get(k) or k[:8]} x{v}' for k, v in b["block_rule"].most_common(5))
    text = (f"opnsense pf filter summary day={day} iface={iface} ({name}){' PARTIAL-DAY' if partial else ''} "
            f"lines={b['lines']} pass={n_pass} block={n_block}. "
            f"top blocked inbound sources: {bs or 'none'}. top blocked inbound ports: {bp or 'none'}. "
            f"top block rules: {br or 'none'}.")[:1800]
    payload = {"type": "filter_day_summary", "day": day, "iface": iface, "iface_name": name,
               "partial_day": partial, "lines": b["lines"], "pass": n_pass, "block": n_block,
               "totals": dict(tot.most_common(40)), "top_block_src": topn(b["block_src"]),
               "top_block_dport": topn(b["block_dport"]), "top_block_rule": topn(b["block_rule"], labels),
               "top_pass_src": topn(b["pass_src"]), "top_pass_dport": topn(b["pass_dport"]),
               "distinct_block_src": len(b["block_src"]), "text": text,
               "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    return text, payload


def ensure_collection(h):
    r = requests.get(f"{base.QDRANT}/collections/{COLL}", headers=h, timeout=30)
    if r.status_code == 200:
        return
    requests.put(f"{base.QDRANT}/collections/{COLL}", headers=h, timeout=30,
                 json={"vectors": {"size": 768, "distance": "Cosine"}}).raise_for_status()
    for fld in ("day", "iface", "type"):
        requests.put(f"{base.QDRANT}/collections/{COLL}/index", headers=h, timeout=30,
                     json={"field_name": fld, "field_schema": "keyword"})


def write(rendered, h):
    ensure_collection(h)
    for i in range(0, len(rendered), 32):
        batch = rendered[i:i + 32]
        vecs = base.embed([t for _, t, _ in batch])
        pts = [{"id": str(uuid.uuid5(NS, key)), "vector": v, "payload": p} for (key, _, p), v in zip(batch, vecs)]
        requests.put(f"{base.QDRANT}/collections/{COLL}/points?wait=true", headers=h,
                     json={"points": pts}, timeout=120).raise_for_status()


def load_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(file, offset, agg_open):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"file": file, "offset": offset,
                   "agg": {f"{d}|{i}": bucket_to_json(b) for (d, i), b in agg_open.items()},
                   "saved": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, f)
    os.replace(tmp, STATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--since", help="YYYYMMDD: ignore older files (testing / limited backfill); ignored once state exists")
    a = ap.parse_args()
    base.take_lock(LOCK)
    try:
        st = load_state()
        files = list_files()
        if not files:
            raise RuntimeError("no filter_*.log on the router")
        if a.since and not st.get("file"):
            files = [f for f in files if f[0] >= f"filter_{a.since}.log"]
        todo = plan(files, st)
        newest = files[-1][0]
        agg = {tuple(k.split("|", 1)): bucket_from_json(v) for k, v in st.get("agg", {}).items()}
        touched = set(agg)
        log("plan", files=[(n, o, s) for n, o, s in todo], dry_run=a.dry_run, carried_buckets=len(agg))
        last_file, last_off, bad = st.get("file"), st.get("offset", 0), 0
        for name, off, size in todo:
            consumed = 0
            for item in base.read_segment(f"{LOGDIR}/{name}", off, size):
                if isinstance(item, int):
                    consumed = item
                    break
                before = len(agg)
                if not feed(agg, item):
                    bad += 1
            touched |= set(agg)
            last_file, last_off = name, off + consumed
            log("file done", file=name, consumed=consumed, buckets=len(agg), bad_lines=bad)
        if not todo:
            log("nothing new")
        newest_day = newest[len("filter_"):len("filter_") + 8]
        newest_day = f"{newest_day[:4]}-{newest_day[4:6]}-{newest_day[6:]}"
        labels = rule_labels()
        rendered = []
        for (day, iface) in sorted(touched):
            b = agg[(day, iface)]
            text, payload = render(day, iface, b, labels, partial=(day >= newest_day))
            rendered.append((f"{day}|{iface}", text, payload))
        for key, text, p in rendered[-6:]:
            log("summary", key=key, lines=p["lines"], block=p["block"], text=text[:200])
        if a.dry_run:
            return
        if rendered:
            write(rendered, base.qheaders())
            log("written", points=len(rendered))
        keep = {k: v for k, v in agg.items() if k[0] >= newest_day}      # finalised past days leave the state
        save_state(last_file, last_off, keep)
        log("state saved", file=last_file, offset=last_off, open_buckets=len(keep))
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
