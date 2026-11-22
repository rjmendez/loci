#!/usr/bin/env python3
"""Post-processing for the ops_logs collection.

  retention   delete points older than --keep-days (default 45)
  secretscan  delete any point whose text still looks like a credential; report patterns only
  freshness   write ops_logs_health.json; log a Windows Application warning if ingest is stale
  digest      daily markdown digest of new/top warning+error signatures, linked to Loci memory
  all         retention + secretscan + digest + freshness
Read-only on log sources; only touches the ops_logs collection and ~/.loci/ops_digests.
"""
import argparse, collections, datetime as dt, json, os, re, subprocess, sys, threading, time, uuid
import requests
import ops_logs_ship as S

H = None
# The scheduled task is killed at 20 minutes with no log line (result 0x41306). The job gives itself less than that
# and says which step it was in, so a stalled Qdrant or Ollama is a diagnosis, not a silent kill.
DEADLINE_S = int(os.environ.get("OPS_POST_DEADLINE_S", "900"))
EMBED_TIMEOUT_S = int(os.environ.get("OPS_POST_EMBED_TIMEOUT_S", "90"))
D = os.path.join(S.HOME, ".loci", "ops_digests")
HEALTH = os.path.join(S.HOME, ".loci", "ops_logs_health.json")
SECRETS = {
    "private-key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    "aws-key": re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b"),
    "github": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "slack": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "sk-key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "kv-secret": re.compile(r"(?i)(password|passwd|secret|api[_-]?key|token)\s*[=:]\s*(?!\*\*\*)[^\s\"',;]{8,}"),
    "bearer": re.compile(r"(?i)bearer\s+(?!\*\*\*)[A-Za-z0-9._-]{16,}"),
}


def _req(method, url, tries=3, **kw):
    """requests.request with bounded retries for a dropped or stalled connection. Every Qdrant call in this job is
    idempotent (scroll, delete by id, upsert by a fixed id, a count, a search), so repeating one is safe. A reset
    from the k3s NodePort failed the whole secretscan step once (2026-10-05 19:14); the script itself finishes in seconds."""
    last = None
    for attempt in range(tries):
        try:
            return requests.request(method, url, **kw)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = exc
            if attempt < tries - 1:
                time.sleep(2 * (attempt + 1))
    raise last


def scroll(fields=True, flt=None, colls=None):
    for coll in colls or (S.COLL, S.AGENT_COLL):
        off = None
        while True:
            body = {"limit": 500, "with_payload": fields, "with_vector": False}
            if off:
                body["offset"] = off
            if flt:
                body["filter"] = flt
            r = _req("POST", f"{S.QDRANT}/collections/{coll}/points/scroll", headers=H, json=body, timeout=120)
            r.raise_for_status()
            res = r.json()["result"]
            for p in res["points"]:
                p["_c"] = coll
                yield p
            off = res.get("next_page_offset")
            if not off:
                break


def delete(pts):
    """pts: iterable of scrolled points (need id and _c)."""
    by = collections.defaultdict(list)
    for p in pts:
        by[p["_c"]].append(p["id"])
    for coll, ids in by.items():
        for i in range(0, len(ids), 500):
            _req("POST", f"{S.QDRANT}/collections/{coll}/points/delete?wait=true", headers=H,
                 json={"points": ids[i:i + 500]}, timeout=120).raise_for_status()


def retention(keep):
    cutoff = (dt.date.today() - dt.timedelta(days=keep)).isoformat()
    old = [p for p in scroll() if p["payload"].get("type") == "ops_log" and p["payload"].get("day", "9") < cutoff]
    delete(old)
    print(f"retention: deleted {len(old)} points older than {cutoff}")
    return len(old)


def secretscan():
    hits, ids = collections.Counter(), []   # ids holds scrolled points
    for p in scroll():
        t = p["payload"].get("text", "") + " " + p["payload"].get("sample", "")
        for name, rx in SECRETS.items():
            if rx.search(t):
                hits[name] += 1
                ids.append(p)
                break
    delete(ids)
    print(f"secretscan: removed {len(ids)} points; patterns: {dict(hits)}")
    return dict(hits)


def freshness(max_age_min=45):
    age = (dt.datetime.now().timestamp() - os.path.getmtime(S.STATE)) / 60 if os.path.exists(S.STATE) else 1e9
    pts = _req("GET", f"{S.QDRANT}/collections/{S.COLL}", headers=H, timeout=30).json()["result"]["points_count"]
    ti = subprocess.run([S.PSEXE, "-NoProfile", "-Command",
                         "(Get-ScheduledTaskInfo 'Loci-OpsLogs-Ship') | Select LastTaskResult,@{n='t';e={$_.LastRunTime.ToString('o')}} | ConvertTo-Json -Compress"],
                        capture_output=True, text=True).stdout.strip()
    info = json.loads(ti) if ti else {}
    # 267009 (SCHED_S_TASK_RUNNING) is the ship task being mid-run when this looks, not a failure: counting it as one
    # raised a false stale alarm (and a Windows warning event) whenever the two coincided.
    ok = age <= max_age_min and info.get("LastTaskResult", 0) in (0, 267009)
    json.dump({"ok": ok, "state_age_min": round(age, 1), "points": pts, "task": info,
               "checked": dt.datetime.now().isoformat(timespec="seconds")}, open(HEALTH, "w"), indent=1)
    print(f"freshness: ok={ok} age={age:.0f}min points={pts} task={info}")
    if not ok:   # Application log is itself ingested by the shipper, so staleness becomes searchable
        subprocess.run(["eventcreate", "/T", "WARNING", "/ID", "901", "/L", "APPLICATION", "/SO", "LociOpsLogs",
                        "/D", f"ops_logs ingest stale: state age {age:.0f} min, last task result {info.get('LastTaskResult')}"],
                       capture_output=True)
    return ok


def link(texts):
    """Nearest Loci memory finding for each signature (score >= 0.68), else None."""
    out = []
    try:
        vecs = S.embed([t[:1200] for t in texts], timeout=EMBED_TIMEOUT_S)
    except Exception:
        return [None] * len(texts)
    for v in vecs:
        r = _req("POST", f"{S.QDRANT}/collections/loci_memory/points/search", headers=H, timeout=60,
                          json={"vector": v, "limit": 1, "with_payload": ["investigation_id", "content", "text"]})
        best = r.json().get("result", [None])[0] if r.ok else None
        out.append(best if best and best["score"] >= 0.68 else None)
    return out


def digest(day=None):
    if not day:   # morning runs summarise the day that just ended
        d0 = dt.date.today()
        day = (d0 - dt.timedelta(days=1) if dt.datetime.now().hour < 12 else d0).isoformat()
    prior_from = (dt.date.fromisoformat(day) - dt.timedelta(days=7)).isoformat()
    cur, prior = {}, set()
    for p in scroll(colls=[S.COLL]):
        pl = p["payload"]
        if pl.get("type") != "ops_log" or pl.get("severity") not in ("warning", "error", "critical"):
            continue
        key = (pl["source"], S.signature(pl["sample"]))
        if pl["day"] == day:
            cur[key] = pl
        elif prior_from <= pl["day"] < day:
            prior.add(key)
    new = sorted((v for k, v in cur.items() if k not in prior), key=lambda v: -v["count"])
    top = sorted(cur.values(), key=lambda v: (-{"critical": 3, "error": 2, "warning": 1}[v["severity"]], -v["count"]))
    pick = {id(v): v for v in new[:12] + top[:12]}.values()
    pick = list(pick)
    links = link([v["sample"] for v in pick])
    bysrc = collections.Counter()
    for v in cur.values():
        bysrc[(v["source"], v["severity"])] += v["count"]
    L = [f"# Ops log digest {day}", "",
         f"{len(cur)} warning+ signatures today, {len(new)} not seen in the previous 7 days.", "",
         "## Volume by source", ""]
    L += [f"- {s} ({sev}): {c}" for (s, sev), c in bysrc.most_common(15)]
    L += ["", "## New signatures (by count)", ""]
    lk = {id(v): l for v, l in zip(pick, links)}
    for v in new[:12]:
        L.append(f"- x{v['count']} [{v['source']}] {v['sample'][:200]}")
        if lk.get(id(v)):
            L.append(f"  - related memory: inv={lk[id(v)]['payload'].get('investigation_id')} score={lk[id(v)]['score']:.2f}")
    L += ["", "## Top errors/warnings", ""]
    for v in top[:12]:
        L.append(f"- {v['severity']} x{v['count']} [{v['source']}] {v['sample'][:200]}")
        if lk.get(id(v)):
            L.append(f"  - related memory: inv={lk[id(v)]['payload'].get('investigation_id')} score={lk[id(v)]['score']:.2f}")
    md = "\n".join(L)
    os.makedirs(D, exist_ok=True)
    open(os.path.join(D, f"{day}.md"), "w", encoding="utf-8").write(md)
    vec = S.embed([md[:3000]], timeout=EMBED_TIMEOUT_S)[0]
    _req("PUT", f"{S.QDRANT}/collections/{S.COLL}/points?wait=true", headers=H, timeout=60, json={"points": [{
        "id": str(uuid.uuid5(S.NS, f"digest|{day}")), "vector": vec,
        "payload": {"type": "ops_digest", "source": "ops-digest", "host": "ALL", "day": day,
                    "severity": "info", "count": len(cur), "text": md[:6000], "sample": md[:300]}}]}).raise_for_status()
    print(f"digest {day}: {len(cur)} signatures, {len(new)} new -> {os.path.join(D, day + '.md')}")


def _stamp(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run_steps(steps, deadline_s=None, exit_fn=os._exit):
    """Run ``[(name, callable)]`` one after another. A step that raises is reported and the next one still runs; the
    return value is the list of failed step names. If the whole job outlives ``deadline_s`` it names the step it was
    in and exits 3 (``exit_fn`` is injectable for tests)."""
    state = {"step": "(starting)"}
    timer = None
    if deadline_s:
        def fire():
            _stamp(f"post: DEADLINE {deadline_s}s exceeded during step '{state['step']}'; exiting")
            exit_fn(3)
        timer = threading.Timer(deadline_s, fire)
        timer.daemon = True
        timer.start()
    failed = []
    try:
        for name, fn in steps:
            state["step"] = name
            started = time.monotonic()
            _stamp(f"{name}: start")
            try:
                fn()
                _stamp(f"{name}: done in {time.monotonic() - started:.1f}s")
            except Exception as exc:
                failed.append(name)
                _stamp(f"{name}: FAILED after {time.monotonic() - started:.1f}s: {type(exc).__name__}: {str(exc)[:200]}")
    finally:
        if timer:
            timer.cancel()
    _stamp("post: " + ("all steps ok" if not failed else "failed steps: " + ", ".join(failed)))
    return failed


def main():
    global H
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["retention", "secretscan", "freshness", "digest", "all"])
    ap.add_argument("--keep-days", type=int, default=45)
    ap.add_argument("--day")
    a = ap.parse_args()
    H = S.qheaders()
    steps = []
    if a.cmd in ("retention", "all"):
        steps.append(("retention", lambda: retention(a.keep_days)))
    if a.cmd in ("secretscan", "all"):
        steps.append(("secretscan", secretscan))
    if a.cmd in ("digest", "all"):
        steps.append(("digest", lambda: digest(a.day)))
    if a.cmd in ("freshness", "all"):
        steps.append(("freshness", freshness))
    sys.exit(1 if run_steps(steps, DEADLINE_S) else 0)


if __name__ == "__main__":
    main()
