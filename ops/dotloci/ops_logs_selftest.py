#!/usr/bin/env python3
"""Known-answer self-test for the ops_logs RAG use case.

Runs real rag_context_search calls through Loci and checks:
  - recall: each query's top results contain the expected text (static cases must pass;
    dynamic cases depend on currently-live errors and only warn)
  - separation: agent tool errors do not leak into ops_logs results
  - coverage: key sources have points from today or yesterday (sources that cannot exist on this machine are
    reported as unavailable, using the shipper's own health file, instead of failing for ever)
  - the shipper pipeline and the Qdrant backup are healthy

The checks run CONCURRENTLY, each under a per-check deadline and all under an overall deadline, so a slow Loci
(a CPU reranker made one rag_context_search take 60-136 s) can no longer push the run past the driver's 280 s.
Every result has a status:
  ok           passed
  slow         passed, but a search took >= SLOW_S seconds (reported with the measured seconds; not a failure)
  fail         the known answer was wrong / a leak / a required source has 0 points / the pipeline or backup is bad
  unavailable  the check could not run (Loci or Qdrant unreachable, source cannot exist here); reason in `detail`
  timeout      no answer before the deadline; reported with seconds; NOT a failure
Only a REAL failure (`fail` on a static or coverage check) makes the exit code 1, writes the Windows event and flips
ok=false. Timeouts and unavailable results are listed under `degraded` in ops_selftest.json and in the summary line.
Writes ops_selftest.json (with per-check timings). Run after ingest (the Post task runs it daily; the Reflect task
runs it every cycle).
"""
import datetime as dt, json, os, re, subprocess, sys, threading, time
import requests
import ops_logs_ship as S
import ops_loci_client as L

OUT = os.path.join(S.HOME, ".loci", "ops_selftest.json")
BACKUP_HEALTH = os.path.join(S.HOME, ".loci", "qdrant_backup_health.json")
PIPELINE_HEALTH = os.path.join(S.HOME, ".loci", "ops_logs_health.json")
SHIP_HEALTH = S.SHIP_HEALTH

TOTAL_S = float(os.environ.get("LOCI_SELFTEST_TOTAL_S", "150"))      # overall deadline (driver allows 280)
CHECK_S = float(os.environ.get("LOCI_SELFTEST_CHECK_S", "140"))      # per-check deadline, counted from its start
SLOW_S = float(os.environ.get("LOCI_SELFTEST_SLOW_S", "30"))         # a passing search slower than this is `slow`
# Searches in flight at once. 1 by default: while the reranker runs on the CPU a search is CPU-bound, so measured on
# 2026-10-19 three concurrent searches all missed a 140 s deadline that one alone nearly meets, and the abandoned calls keep
# running inside Loci. Raise it (Loci has 4 tool workers) once searches take seconds.
SEARCH_PARALLEL = int(os.environ.get("LOCI_SELFTEST_PARALLEL", "1"))
SHIP_HEALTH_MAX_AGE_H = 2.0

CASES = [
    # (id, query, collections, expected regex, kind)
    ("cuda-oom", "flybrain GPU out of memory job failed", ["ops_logs"], r"CUDA out of memory|OutOfMemory", "static"),
    ("k3s-pgrep", "k3s ExecSync failed container pgrep not found", ["ops_logs"], r"ExecSync|pgrep", "static"),
    # A generic "crash loop" query legitimately matches several real events (Cloudflared, WSL service), so the one entry
    # this expects landed in the top 5 only about half the time (6/12 repeats). Asking for what that entry says: 12/12.
    ("vague-service-loop", "service ran for less than the minimum time and keeps restarting", ["ops_logs"], r"nssm|OpenClaw|ran for less than", "dynamic"),
    ("etl-denied", "why is the daily ETL not launching tasks for pixel devices", ["ops_logs"], r"AccessDenied|ecs:RunTask", "dynamic"),
    ("failed-logon", "failed logon attempts Windows security", ["ops_logs"], r"4625|failed to log on", "dynamic"),
    ("agent-errors-own-collection", "Restart-Service cannot be stopped service", ["ops_logs_agent"], r"Restart-Service|cannot be stopped", "static"),
]
LEAK = ("agent-leak", "Restart-Service cannot be stopped service", ["ops_logs"], r"src=claude-tool-errors")
# (source with points in Qdrant, shipper collector that produces it, mode)
#   required    : must have points from today or yesterday; 0 is a real failure
#   conditional : required only while the shipper reports the collector `ok`; if the shipper says it is unavailable
#                 (file or tool absent on the running WSL distro) the check is `unavailable` with that reason.
# wsl-tmux-watchdog (/var/log/tmux-watchdog.log) and k3s-pod-dama (kubectl) do not exist on UbuntuCap, the distro
# that has run since the 2026-10-14 restore, so they must not be `required`.
COVER = [
    ("wsl-syslog", "wsl-syslog", "required"),
    ("wsl-tmux-watchdog", "wsl-tmux-watchdog", "conditional"),
    ("win-eventlog-application", None, "required"),
    ("k3s-pod-dama", "k3s-pods", "conditional"),
]
SEARCH_ARGS = {"limit": 5, "budget_chars": 5000, "mode": "compact"}


def _expand_arg():
    v = os.environ.get("LOCI_SELFTEST_EXPAND")   # unset = the server default; "0"/"1" force query expansion off/on
    return {} if v is None else {"expand_query": v not in ("0", "false", "False", "")}


def search(q, colls):
    r = L.call("rag_context_search", query=q, collections=colls, **SEARCH_ARGS, **_expand_arg())
    return r.get("context", "") if isinstance(r, dict) else str(r)


# ---------------------------------------------------------------- results and classification
def leaf_exceptions(exc):
    if isinstance(exc, BaseExceptionGroup):
        out = []
        for e in exc.exceptions:
            out.extend(leaf_exceptions(e))
        return out
    return [exc]


def classify_exception(exc):
    """(status, detail) for an exception raised by a check. Only timeouts and 'cannot reach it' are non-failures."""
    leaves = leaf_exceptions(exc)
    for e in leaves:
        name = type(e).__name__
        if isinstance(e, getattr(L, "LociUnreachable", ())) or isinstance(e, (requests.ConnectionError, ConnectionError)) \
                or name in ("ConnectError", "ConnectTimeout"):
            return "unavailable", f"{name}: {e}"[:200]
    for e in leaves:
        if isinstance(e, (TimeoutError, subprocess.TimeoutExpired, requests.Timeout)):
            return "timeout", f"{type(e).__name__}: {e}"[:200]
    e = leaves[0]
    return "fail", f"{type(e).__name__}: {e}"[:200]


def is_real_failure(r):
    return r["status"] == "fail" and r["kind"] != "dynamic"


# ---------------------------------------------------------------- the checks
def check_recall(q, colls, rx):
    def run(clock):
        t0 = clock()
        ctx = search(q, colls)
        took = clock() - t0
        if not re.search(rx, ctx):
            return {"status": "fail", "detail": "expected text not in the top results"}
        return {"status": "slow" if took >= SLOW_S else "ok"}
    return run


def check_leak(q, colls, rx):
    def run(clock):
        t0 = clock()
        ctx = search(q, colls)
        took = clock() - t0
        if re.search(rx, ctx):
            return {"status": "fail", "detail": "agent tool errors leaked into ops_logs results"}
        return {"status": "slow" if took >= SLOW_S else "ok"}
    return run


def read_ship_health(path=None, now=None):
    """(doc, None) from the shipper's health file, or (None, reason) when absent/stale/unreadable."""
    path = path or SHIP_HEALTH
    try:
        doc = json.load(open(path, encoding="utf-8"))
        age_h = ((now or dt.datetime.now()) - dt.datetime.fromisoformat(doc["ts"])).total_seconds() / 3600
    except Exception as e:
        return None, f"no shipper health ({type(e).__name__})"
    if age_h > SHIP_HEALTH_MAX_AGE_H:
        return None, f"shipper health is {age_h:.1f} h old"
    return doc, None


def check_coverage(src, collector, mode, days, headers):
    def run(clock):
        r = requests.post(f"{S.QDRANT}/collections/{S.COLL}/points/count", headers=headers, timeout=30, json={"exact": True, "filter": {
            "must": [{"key": "source", "match": {"value": src}}, {"key": "day", "match": {"any": days}}]}})
        if not r.ok:
            return {"status": "fail", "detail": f"qdrant count HTTP {r.status_code}", "points": -1}
        n = r.json()["result"]["count"]
        if n > 0:
            return {"status": "ok", "points": n}
        doc, why = read_ship_health()
        ship = (doc or {}).get("sources", {}).get(collector) if collector else None
        wsl = (doc or {}).get("wsl", {})
        where = f"; shipper reads WSL distro {wsl.get('distro')} ({wsl.get('how')})" if wsl else ""
        if mode == "conditional":
            if doc is None:
                return {"status": "unavailable", "points": 0, "detail": f"0 points and {why}"}
            if ship is None:
                return {"status": "unavailable", "points": 0, "detail": f"shipper did not run collector {collector}{where}"}
            if ship["status"] == "unavailable":
                return {"status": "unavailable", "points": 0, "detail": f"cannot ship here: {ship.get('detail')}{where}"}
            return {"status": "fail", "points": 0, "detail": f"0 points but shipper collector {collector} is {ship['status']}{where}"}
        extra = f"; shipper: {ship['status']} {ship.get('detail', '')}".rstrip() if ship else ""
        return {"status": "fail", "points": 0, "detail": f"required source has 0 points for {days}{extra}{where}"}
    return run


def check_pipeline(clock):
    subprocess.run([sys.executable, os.path.join(S.HOME, ".loci", "ops_logs_post.py"), "freshness"], capture_output=True, timeout=100)
    hl = json.load(open(PIPELINE_HEALTH))
    return {"status": "ok" if hl.get("ok") else "fail", "detail": hl.get("task")}


def check_backup(clock):
    bk = json.load(open(BACKUP_HEALTH))
    age_h = (dt.datetime.now() - dt.datetime.fromisoformat(bk["ts"])).total_seconds() / 3600
    ok = bool(bk.get("ok")) and age_h < 36
    errs = bk.get("errors", [])
    detail = f"{bk.get('backed_up')} collections, {age_h:.1f} h old, {len(errs)} errors"
    if errs:
        detail += f"; first: {str(errs[0])[:120]}"
    return {"status": "ok" if ok else "fail", "detail": detail}


def build_checks(today=None):
    today = today or dt.date.today()
    days = [today.isoformat(), (today - dt.timedelta(days=1)).isoformat()]
    checks = []
    searches = [{"id": cid, "kind": kind, "group": "search", "fn": check_recall(q, colls, rx)} for cid, q, colls, rx, kind in CASES]
    searches.append({"id": LEAK[0], "kind": "static", "group": "search", "fn": check_leak(LEAK[1], LEAK[2], LEAK[3])})
    checks.extend(sorted(searches, key=lambda c: c["kind"] != "static"))   # static first: they matter most if the deadline bites
    headers = S.qheaders()
    for src, collector, mode in COVER:
        checks.append({"id": f"coverage:{src}", "kind": "coverage", "group": None,
                       "fn": check_coverage(src, collector, mode, days, headers)})
    checks.append({"id": "pipeline-health", "kind": "coverage", "group": None, "fn": check_pipeline})
    checks.append({"id": "qdrant-backup", "kind": "coverage", "group": None, "fn": check_backup})
    return checks


# ---------------------------------------------------------------- the concurrent runner
def run_checks(checks, per_check_s, total_s, search_parallel=3, clock=time.monotonic, wait=time.sleep, poll_s=0.05):
    """Run every check in its own daemon thread (checks with group 'search' share `search_parallel` slots).
    Returns results in the input order. A check that has not answered by min(start + per_check_s, overall deadline)
    is reported as `timeout` and abandoned; the daemon thread cannot hold the process open."""
    slots = threading.Semaphore(max(1, search_parallel))
    state = {c["id"]: {"started": None, "ended": None, "res": None} for c in checks}

    def work(c):
        st = state[c["id"]]
        gate = slots if c.get("group") == "search" else None
        if gate:
            gate.acquire()
        st["started"] = clock()
        try:
            res = c["fn"](clock)
        except BaseException as e:   # noqa: BLE001 - a check must never take the runner down
            status, detail = classify_exception(e)
            res = {"status": status, "detail": detail}
        st["ended"] = clock()
        st["res"] = res
        if gate:
            gate.release()

    t0 = clock()
    for c in checks:
        threading.Thread(target=work, args=(c,), daemon=True, name="check-" + c["id"]).start()
    out = {}
    while len(out) < len(checks):
        now = clock()
        for c in checks:
            cid = c["id"]
            if cid in out:
                continue
            st = state[cid]
            if st["res"] is not None:
                out[cid] = dict(st["res"], seconds=round(st["ended"] - st["started"], 1))
            elif st["started"] is not None and now - st["started"] >= per_check_s:
                out[cid] = {"status": "timeout", "seconds": round(now - st["started"], 1),
                            "detail": f"no answer within the {per_check_s:.0f} s per-check deadline"}
            elif now - t0 >= total_s:
                started = st["started"] is not None
                out[cid] = {"status": "timeout", "seconds": round(now - st["started"], 1) if started else None,
                            "detail": f"{'still running' if started else 'not started'} at the {total_s:.0f} s overall deadline"}
        if len(out) < len(checks):
            wait(poll_s)
    return [dict(id=c["id"], kind=c["kind"], **out[c["id"]]) for c in checks]


def finalize(results):
    for r in results:
        # legacy boolean: True passed, False failed, None = no verdict (timeout / unavailable)
        r["ok"] = True if r["status"] in ("ok", "slow") else (False if r["status"] == "fail" else None)
    real = [r["id"] for r in results if is_real_failure(r)]
    degraded = [f'{r["id"]}:{r["status"]}' for r in results if r["status"] in ("timeout", "unavailable")]
    slow = [f'{r["id"]}:{r["seconds"]}s' for r in results if r["status"] == "slow"]
    return real, degraded, slow


def emit_event(bad):
    subprocess.run(["eventcreate", "/T", "WARNING", "/ID", "902", "/L", "APPLICATION", "/SO", "LociOpsLogs",
                    "/D", f"ops_logs self-test failed: {bad}"], capture_output=True)


def write_json(doc, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)


def main(checks=None, out=None, emit=None, clock=time.monotonic, wait=time.sleep):
    t_start = clock()
    checks = build_checks() if checks is None else checks
    results = run_checks(checks, CHECK_S, TOTAL_S, SEARCH_PARALLEL, clock=clock, wait=wait)
    real, degraded, slow = finalize(results)
    doc = {"ok": not real, "ts": dt.datetime.now().isoformat(timespec="seconds"),
           "elapsed_s": round(clock() - t_start, 1), "degraded": degraded, "slow": slow, "results": results}
    write_json(doc, out or OUT)
    passed = sum(1 for r in results if r["status"] in ("ok", "slow"))
    print(("PASS" if not real else "FAIL"), f"{passed}/{len(results)} ok; failing: {real}; slow: {slow}; degraded: {degraded}")
    if real:
        (emit or emit_event)(real)
    return 1 if real else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)   # abandoned (timed-out) check threads must not hold the process open
