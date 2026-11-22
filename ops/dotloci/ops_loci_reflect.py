#!/usr/bin/env python3
"""Self-reflection driver for Loci, covering the Copilot reflection loop AND the ops_logs RAG use case.

Each run (scheduled every 30 min as Loci-OpsLogs-Reflect):
  1. reflection loop: seed when the queue is empty and >24h since last seed, then tick (bounded);
     if backlog > 200 tick harder (the tier-1 reflex from the self-model design)
  2. ops_logs self-test (ops_logs_selftest.py); store a finding in investigation `ops-logs-rag`
     when pass/fail flips, and once a day with the digest's top new signatures
  3. once a day: investigation_reflect on ops-logs-rag so the loop critiques its own findings
State: ~/.loci/ops_reflect_state.json. Never deletes or retracts anything.
"""
import datetime as dt, json, os, re, subprocess, sys
import ops_loci_client as L

HOME = os.path.expanduser("~")
ST = os.path.join(HOME, ".loci", "ops_reflect_state.json")
SELFTEST_JSON = os.path.join(HOME, ".loci", "ops_selftest.json")
INV = "ops-logs-rag"
PY = sys.executable
SELFTEST_TIMEOUT_S = int(os.environ.get("OPS_SELFTEST_TIMEOUT_S", "280"))   # the self-test limits itself to ~150 s; this is the net
DEGRADED_ALERT_RUNS = 6      # consecutive runs (3 h) of timeouts / unavailable checks before a finding is stored
TICK_MIN = 5


def load():
    try:
        return json.load(open(ST))
    except Exception:
        return {}


def save(st, path=None):
    """Atomic state write. Called after every step, so a run killed by the Task Scheduler limit keeps what it did."""
    path = path or ST
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, path)


def now():
    return dt.datetime.now(dt.timezone.utc)


def hours_since(iso):
    return 1e9 if not iso else (now() - dt.datetime.fromisoformat(iso)).total_seconds() / 3600


def store(kind, text, source, conf="medium", tags="ops_logs,reflection"):
    return L.call("investigation_store", investigation_id=INV, finding_type=kind, text=text, source=source,
                  confidence=conf, tags=tags)


def capture(tick):
    """Keep a scrubbed copy of each processed item's inputs so a person can label them
    (hillclimb.py label). Needs a Loci build with hillclimb.capture_observations; otherwise a no-op."""
    try:
        sys.path.insert(0, os.path.join(HOME, "loci-deploy", "mcp"))
        import hillclimb
        fn = getattr(hillclimb, "capture_observations", None)
        if fn:
            n = fn((tick or {}).get("batch", []))
            if n:
                print(f"hillclimb: captured {n} new observation(s)")
    except Exception as e:
        print(f"hillclimb capture skipped: {type(e).__name__}: {e}")


def reflection_loop(st):
    s = L.call("reflection_loop_status", queue_preview=0)
    q = s.get("queue_size", 0)
    if q == 0 and hours_since(st.get("last_seed")) > 24:
        L.call("reflection_loop_seed", process_logs_limit=400, session_events_limit=1000)
        st["last_seed"] = now().isoformat()
        q = L.call("reflection_loop_status", queue_preview=0).get("queue_size", 0)
        print(f"reflection: seeded, queue={q}")
    # After the Windows flock fix (PR #441) a tick is ~1 s/item; before it, ~25-30 s/item (every lock
    # release cost ~10 s) and 20 items always blew the client timeout. Backlog drains at 60 per run.
    full = 60 if q > 200 else 15
    n = min(full, int(st.get("tick_n") or full))   # shrunk after a tick outlived the client ceiling (Loci slow)
    if q:
        try:
            t = L.call("reflection_loop_tick", max_items=n, max_lines_per_file=4000)
        except TimeoutError:
            st["tick_n"] = max(TICK_MIN, n // 2)
            st["queue"] = q
            raise
        st.pop("tick_n", None)
        print(f"reflection: tick max_items={n} -> {str(t)[:200]}")
        capture(t)
    st["queue"] = q


SUMMARY = re.compile(r"^(PASS|FAIL) \d+/\d+ ok")


def selftest(st, run=subprocess.run, call_store=None):
    """Run the self-test. Exit 0 = no real failure (timeouts/unavailable checks are degraded, not failed); 1 = a real
    failure. A run that printed no PASS/FAIL line crashed or was killed: it says nothing about health, so it never
    flips selftest_ok."""
    call_store = call_store or store
    try:
        r = run([PY, os.path.join(HOME, ".loci", "ops_logs_selftest.py")], capture_output=True, text=True,
                cwd=os.path.join(HOME, ".loci"), timeout=SELFTEST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        st["selftest_timeouts"] = int(st.get("selftest_timeouts", 0)) + 1
        line = f"selftest KILLED after {SELFTEST_TIMEOUT_S} s (no verdict)"
        print("selftest:", line)
        return line
    lines = (r.stdout or "").strip().splitlines()
    line = lines[-1] if lines else "no output"
    if not SUMMARY.match(line):
        tail = ((r.stderr or "").strip().splitlines() or ["no stderr"])[-1][:200]
        line = f"selftest CRASHED rc={r.returncode} (no verdict): {tail}"
        print("selftest:", line)
        return line
    ok = r.returncode == 0
    print("selftest:", line)
    if st.get("selftest_ok") is not None and st["selftest_ok"] != ok:
        call_store("observed" if ok else "gap",
                   f"ops_logs self-test flipped to {'PASS' if ok else 'FAIL'} at {now().isoformat(timespec='seconds')}: {line}",
                   "ops_logs_selftest.py", "high", "ops_logs,selftest")
    st["selftest_ok"] = ok
    st["selftest_timeouts"] = 0
    track_degraded(st, call_store)
    return line


def track_degraded(st, call_store=None, path=None):
    """Timeouts / unavailable checks never fail the self-test, so a Loci that is permanently slow or a source that
    is permanently gone would be invisible. After DEGRADED_ALERT_RUNS consecutive degraded runs store one finding;
    store one more when it clears."""
    call_store = call_store or store
    try:
        doc = json.load(open(path or SELFTEST_JSON, encoding="utf-8"))
        degraded = list(doc.get("degraded") or [])
    except Exception:
        return
    if degraded:
        st["degraded_runs"] = int(st.get("degraded_runs", 0)) + 1
        if st["degraded_runs"] >= DEGRADED_ALERT_RUNS and not st.get("degraded_reported"):
            call_store("gap", f"ops_logs self-test has been degraded for {st['degraded_runs']} consecutive runs: {degraded}",
                       "ops_logs_selftest.py", "medium", "ops_logs,selftest,degraded")
            st["degraded_reported"] = True
    else:
        if st.get("degraded_reported"):
            call_store("observed", f"ops_logs self-test is no longer degraded (was {st.get('degraded_runs')} runs)",
                       "ops_logs_selftest.py", "medium", "ops_logs,selftest,degraded")
        st["degraded_runs"] = 0
        st["degraded_reported"] = False


def daily(st, line):
    if hours_since(st.get("last_daily")) < 24:
        return
    d = os.path.join(HOME, ".loci", "ops_digests")
    top = ""
    try:
        f = sorted(os.listdir(d))[-1]
        md = open(os.path.join(d, f), encoding="utf-8").read()
        top = md.split("## New signatures")[1][:1500] if "## New signatures" in md else md[:1500]
    except Exception:
        pass
    store("observed", f"Daily ops_logs reflection {now().date()}: self-test {line}. Latest digest new signatures:{top}",
          "ops_logs_digest+selftest", "medium", "ops_logs,daily")
    r = L.call("investigation_reflect", investigation_id=INV)
    crit = (r.get("self_critique") or "")[:800] if isinstance(r, dict) else ""
    print("reflect: gaps", len((r or {}).get("gaps", [])) if isinstance(r, dict) else "?", "critique:", crit[:200])
    st["last_daily"] = now().isoformat()


def step(name, fn, st):
    """Run one step; an exception is logged and never stops the next step; state is saved either way."""
    try:
        return fn()
    except Exception as e:
        print(f"{name} FAILED {type(e).__name__}: {e}")
    finally:
        save(st)


def main():
    st = load()
    step("reflection_loop", lambda: reflection_loop(st), st)
    line = step("selftest", lambda: selftest(st), st) or "selftest n/a (step failed)"
    step("daily", lambda: daily(st, line), st)   # independent of the self-test: it used to be skipped whenever that raised


if __name__ == "__main__":
    main()
