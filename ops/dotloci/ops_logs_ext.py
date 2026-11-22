"""Extra collectors for ops_logs_ship: Loci/flybrain logs, journald user units,
k3s pod logs + warning events, Claude tool errors, extra Windows event logs,
rotated-log backfill. All read-only. Imported lazily by ops_logs_ship.collectors()."""
import datetime as dt, glob, gzip, json, os, re, subprocess, sys

import ops_logs_ship as S

WSLHOME = S.WSLHOME
FLYBRAIN_LOGS = r"F:\.flybrain\logs"
JOURNAL_UNITS = ["loci-mcp", "loci-a2a-pr378", "mrpink-a2a", "loci-gpu-load", "hermes-grounding-mrpink",
                 "mrpink-context-bridge", "flybrain-watch", "flybrain-disk-guard", "flybrain-slot-doctor"]
S.FLOOR |= {"wsl-journal", "k3s-pod", "flybrain-run-log"}
JIDENT = re.compile(r"^\S+ \S+ ([^\s\[:]+)(?:\[\d+\])?:")


def now_iso():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def wsl(script, timeout=300):
    r = subprocess.run(S.wsl_argv(), input=script.replace("\r\n", "\n").encode(),
                       capture_output=True, timeout=timeout)   # bytes in: Windows text mode would add CRs
    return r.stdout.decode("utf-8", "replace")


def ship_journal(agg, state, new_state, full):
    key = "journal:since"
    since = state.get(key) or (dt.datetime.now() - dt.timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    started = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    units = " ".join(f"-u {u}" for u in JOURNAL_UNITS)
    out = wsl(f'journalctl --user {units} --since "{since}" -o short-iso --no-pager 2>/dev/null | cut -c1-1500')
    n = 0
    for ln in out.splitlines():
        m = S.TS.match(ln)
        if not m:
            continue
        ident = (JIDENT.match(ln) or [None, "unknown"])[1]
        agg.add(f"wsl-journal-{ident}", "WSL", "journald:user", m.group(1), f"{m.group(1)}T{m.group(2)}", ln)
        n += 1
    new_state[key] = started
    return n


POD_SCRIPT = r'''
SINCE="%s"
kubectl get pods -A --no-headers -o custom-columns=NS:.metadata.namespace,POD:.metadata.name 2>/dev/null | while read ns pod; do
  kubectl logs -n "$ns" "$pod" --all-containers --timestamps --since-time="$SINCE" --tail=400 </dev/null 2>/dev/null | cut -c1-1500 | sed "s|^|$ns $pod |"
done
'''


def ship_k3s_pods(agg, state, new_state, full):
    key = "k3s:pods:since"
    S.require_wsl_command("kubectl")   # UbuntuCap has no k3s: say so (SourceUnavailable) instead of shipping 0 forever
    since = state.get(key) or (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    started = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = wsl(POD_SCRIPT % since, timeout=500)
    n = 0
    for ln in out.splitlines():
        parts = ln.split(" ", 3)
        if len(parts) < 4:
            continue
        ns, pod, ts, text = parts
        m = S.TS.match(ts)
        if not m:
            continue
        pod_base = re.sub(r"-[a-z0-9]{5,10}(-[a-z0-9]{5})?$", "", pod)
        agg.add(f"k3s-pod-{ns}", "K3S", f"{ns}/{pod_base}", m.group(1), f"{m.group(1)}T{m.group(2)}", f"{pod_base}: {text}")
        n += 1
    new_state[key] = started
    return n


def ship_k3s_events(agg, state, new_state, full):
    key = "k3s:events:since"
    S.require_wsl_command("kubectl")
    since = state.get(key, "")
    out = wsl("kubectl get events -A --field-selector type=Warning -o json 2>/dev/null")
    started = now_iso() + "Z"
    n = 0
    try:
        items = json.loads(out)["items"]
    except Exception:
        return 0
    for ev in items:
        t = ev.get("lastTimestamp") or ev.get("eventTime") or ev.get("metadata", {}).get("creationTimestamp") or ""
        if not t or t <= since:
            continue
        io = ev.get("involvedObject", {})
        name = re.sub(r"-[a-z0-9]{5,10}(-[a-z0-9]{5})?$", "", io.get("name", ""))
        line = f'{io.get("kind")} {io.get("namespace")}/{name} {ev.get("reason")}: {ev.get("message")}'
        agg.add("k3s-events", "K3S", "kubectl:events", t[:10], t[:19], line, "warning")
        n += 1
    new_state[key] = started
    return n


def ship_claude_errors(agg, state, new_state, full):
    base = os.path.join(S.HOME, ".claude", "projects")
    n = 0
    for p in glob.glob(os.path.join(base, "**", "*.jsonl"), recursive=True):
        lines, off = S.read_new(p, state, full)
        new_state[p] = off
        for ln in lines:
            if '"is_error":true' not in ln:
                continue
            try:
                rec = json.loads(ln)
                blocks = rec["message"]["content"]
            except Exception:
                continue
            ts = rec.get("timestamp", "")[:19]
            for b in blocks if isinstance(blocks, list) else []:
                if isinstance(b, dict) and b.get("is_error"):
                    c = b.get("content")
                    if isinstance(c, list):
                        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                    c = re.sub(r"\x1b\[[0-9;]*m", "", str(c)).replace("\r", "")
                    agg.add("claude-tool-errors", "WINDOWS", "claude-code:tool_result", ts[:10], ts,
                            " ".join(c.split())[:700], "error")
                    n += 1
    return n


def ship_flybrain_runlogs(agg, state, new_state, full):
    n = 0
    for p in glob.glob(FLYBRAIN_LOGS + r"\**\*", recursive=True):
        if not (p.endswith(".log") or p.endswith(".jsonl")) or p.endswith(r"watch\events.jsonl"):
            continue
        n += S.ship_textfile(agg, state, new_state, "flybrain-run-log", "WSL", p, full, jsonl=p.endswith(".jsonl"))
    return n


def ship_glob(name, host, pattern, jsonl=False):
    def run(agg, state, new_state, full):
        n = 0
        for p in glob.glob(pattern, recursive=True):
            n += S.ship_textfile(agg, state, new_state, name, host, p, full, jsonl=jsonl)
        return n
    return run


def ship_backfill(agg, state, new_state, full):
    """Rotated logs. syslog.1 is ~870 MB, so only its last 150 MB; syslog .gz archives are skipped."""
    n = 0
    for base, src in (("syslog", "wsl-syslog"), ("kern.log", "wsl-kern"), ("auth.log", "wsl-auth")):
        for p in sorted(glob.glob(S.WSL.replace(chr(92), "/") + "/var/log/" + base + ".*")):
            if base == "syslog" and p.endswith(".gz"):
                continue
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rb") as f:
                if base == "syslog" and not p.endswith(".gz"):
                    f.seek(max(0, os.path.getsize(p) - 150_000_000))
                    f.readline()
                for raw in f:
                    ln = raw.decode("utf-8", "replace")
                    m = S.TS.match(ln)
                    if m:
                        agg.add(src, "WSL", p, m.group(1), f"{m.group(1)}T{m.group(2)}", ln)
                        n += 1
            print(f"  backfill {os.path.basename(p)} done", flush=True)
    return n


CW_SCRIPT = r"""
import boto3, json, sys
s = boto3.Session(profile_name="mrpink-aws", region_name="us-east-1")
L = s.client("logs")
PAT = "?ERROR ?Error ?WARN ?Exception ?Traceback ?FAILED ?Failed ?timeout ?Timeout ?panic ?denied"
for pg in L.get_paginator("describe_log_groups").paginate():
    for g in pg["logGroups"]:
        if not g.get("storedBytes"):
            continue
        n = 0
        try:
            for ev in L.get_paginator("filter_log_events").paginate(logGroupName=g["logGroupName"], startTime=%d, filterPattern=PAT):
                for e in ev["events"]:
                    print(json.dumps({"g": g["logGroupName"], "t": e["timestamp"], "m": e["message"][:1200]}))
                    n += 1
                if n >= 1500:
                    break
        except Exception as ex:
            print(json.dumps({"g": g["logGroupName"], "err": str(ex)[:200]}))
"""


def ship_cloudwatch(agg, state, new_state, full):
    key = "cloudwatch:since_ms"
    started = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    since = state.get(key) or started - 24 * 3600 * 1000
    out = wsl("python3 - <<'PYEOF'\n" + CW_SCRIPT % since + "\nPYEOF\n", timeout=800)
    n = 0
    for ln in out.splitlines():
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if "err" in r:
            print(f"  cloudwatch {r['g']}: {r['err']}")
            continue
        t = dt.datetime.fromtimestamp(r["t"] / 1000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        short = "-".join(r["g"].strip("/").split("/")[-2:]).replace("_", "-")[:40]
        agg.add(f"aws-{short}", "AWS", r["g"], t[:10], t, r["m"].replace("\n", " "))
        n += 1
    new_state[key] = started
    return n


LITELLM_LOGS = (("win-litellm-prod", r"F:\services\litellm\logs\litellm.log"),
                ("win-litellm-dev", r"F:\services\litellm\logs\litellm-dev.log"))
LL_ACCESS = re.compile(r'^INFO:\s+(\S+):\d+ - "(\w+) (\S+) HTTP/[\d.]+" (\d{3})')
LL_LOG = re.compile(r"^(\d\d:\d\d:\d\d) - (LiteLLM[A-Za-z ]*):(WARNING|ERROR|CRITICAL): (\S+) - (.*)$")
LL_EXC = re.compile(r"^(?:[A-Za-z_][\w.]*)(?:Error|Exception|Exit|Interrupt): ")


def ship_litellm(agg, state, new_state, source, path, full):
    """LiteLLM proxy log. WHITELIST only: error HTTP statuses, WARNING+ log records, our own `loci ...` plugin lines,
    supervisor lines and the last line of each traceback. Healthy 2xx access lines and health probes are dropped, and
    so is everything else (tracebacks, banners): provider error bodies can echo prompt text, so messages are cut to 300."""
    if not os.path.exists(path):
        return 0
    lines, new_off = S.read_new(path, state, full)
    new_state[path] = new_off
    day = dt.date.fromtimestamp(os.path.getmtime(path)).isoformat()
    t = "00:00:00"
    for ln in lines:
        m = S.TS.match(ln)
        if m:
            day, t = m.group(1), m.group(2)
            if "[supervisor]" in ln:
                agg.add(source, "WINDOWS", path, day, f"{day}T{t}", "litellm supervisor " + ln[20:320], "info")
            continue
        a = LL_ACCESS.match(ln)
        if a:
            _ip, method, ep, status = a.groups()
            if int(status) >= 400:
                agg.add(source, "WINDOWS", path, day, f"{day}T{t}",
                        f"litellm http {status} {method} {ep.split('?')[0]}", "error" if int(status) >= 500 else "warning")
            continue
        g = LL_LOG.match(ln)
        if g:
            t, comp, level, where, msg = g.groups()
            sev = "warning" if level == "WARNING" else "error"
            agg.add(source, "WINDOWS", path, day, f"{day}T{t}", f"litellm {comp} {level} {where} {msg[:300]}", sev)
            continue
        if ln.startswith("loci "):
            agg.add(source, "WINDOWS", path, day, f"{day}T{t}", "litellm " + ln[:300],
                    "warning" if re.search(r"OPENED|refus|fail|denied", ln, re.I) else "info")
            continue
        if LL_EXC.match(ln):
            agg.add(source, "WINDOWS", path, day, f"{day}T{t}", "litellm exception " + ln[:300], "error")
    return len(lines)



# --- LiteLLM cost-ledger summaries (collector lives, versioned and tested, in the litellm-loci plugin) ---
LEDGER_COLLECTOR = r"F:\services\litellm\polish\litellm-loci\deploy\ops_logs_ledger_collector.py"
LEDGERS = (("win-litellm-ledger-prod", os.path.join(os.path.expanduser("~"), ".loci", "litellm_cost_ledger.jsonl"),
            "litellm_ledger_health.json"),
           ("win-litellm-ledger-dev", r"F:\services\litellm\logs\litellm-dev-cost-ledger.jsonl", None))


def ship_litellm_ledger(agg, state, new_state, source, path, health_name, full):
    """Hourly per-group summaries (aggregates only) and, for prod, the health snapshot Loci can read."""
    import importlib.util
    if not os.path.exists(path):
        return 0
    spec = importlib.util.spec_from_file_location("ops_logs_ledger_collector", LEDGER_COLLECTOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    health = []
    if health_name:
        health = [os.path.join(os.path.expanduser("~"), ".loci", health_name),
                  os.path.join(WSLHOME, ".loci", health_name)]
    return mod.ship_ledger(agg, state, new_state, source, path, full, health_paths=health,
                           write_health="--dry-run" not in sys.argv)


# --- Liftoff / flybrain-racing (added 2026-10-12, owner request: include the Liftoff work in supervision) ---
# WHITELIST only, like ship_litellm: plugin/BepInEx warnings+errors and a few of our own key info lines, the game's
# exceptions / mod-detection / crash-upload lines (cut to 300), and one line per run or scene transition. No raw log.
LIFTOFF_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\Liftoff"
LIFTOFF_BEPINEX = os.path.join(LIFTOFF_DIR, "BepInEx", "LogOutput.log")
LIFTOFF_PLAYER = os.path.join(os.path.expanduser("~"), "AppData", "LocalLow", "LuGus Studios", "Liftoff", "Player.log")
RACING_WORK = r"F:\flybrain-racing-work"
LO_BEP = re.compile(r"^\[(Warning|Error|Fatal|Info)\s*:([^\]]*)\]\s*(.*)$")
LO_KEY = re.compile(r"destroyed|resolved|ready|installed|template|avatar|bind|failed|error|exception", re.I)
LO_PLAYER = re.compile(r"Custom code injection|Uploading Crash Report|Exception|Steamworks is not initialized|EXTERNAL GAME", re.I)


def ship_liftoff_log(agg, state, new_state, source, path, full, kind):
    if not os.path.exists(path):
        return 0
    lines, new_off = S.read_new(path, state, full)
    new_state[path] = new_off
    mt = dt.datetime.fromtimestamp(os.path.getmtime(path))
    day, t = mt.date().isoformat(), mt.strftime("%H:%M:%S")
    for ln in lines:
        if kind == "bepinex":
            m = LO_BEP.match(ln)
            if not m:
                continue
            level, who, msg = m.groups()
            who = who.strip()
            if level == "Info" and not (who == "Flybrain Bridge" and LO_KEY.search(msg)):
                continue
            sev = {"Warning": "warning", "Error": "error", "Fatal": "error"}.get(level, "info")
            agg.add(source, "WINDOWS", path, day, f"{day}T{t}", f"liftoff bepinex {level} [{who}] {msg[:300]}", sev)
        else:
            if LO_PLAYER.search(ln):
                agg.add(source, "WINDOWS", path, day, f"{day}T{t}", "liftoff player " + ln.strip()[:300],
                        "error" if re.search(r"Exception|not initialized", ln) else "warning")
    return len(lines)


def ship_liftoff_runs(agg, state, new_state, full):
    """One summary line per flight run (sweeps, collection flights) and per OBS scene transition."""
    n = 0
    for name, path in (("sweep", os.path.join(RACING_WORK, "runs", "sweep.jsonl")),
                       ("collect", os.path.join(RACING_WORK, "demos", "collect_summary.jsonl"))):
        if not os.path.exists(path):
            continue
        lines, off = S.read_new(path, state, full)
        new_state[path] = off
        day = dt.date.fromtimestamp(os.path.getmtime(path)).isoformat()
        for ln in lines:
            body = ln.split(" ", 1)[1] if (name == "collect" and ln[:1].isdigit() and " " in ln) else ln
            try:
                d = json.loads(body)
            except ValueError:
                continue
            if not isinstance(d, dict) or "policy" not in d:
                continue
            msg = ("liftoff run %s policy=%s act_hz=%s outcome=%s survival_s=%s min_dist_m=%s max_agl_m=%s" % (
                name, d.get("policy"), d.get("act_hz"), d.get("outcome"), d.get("survival_s"),
                d.get("min_dist_to_food_m"), d.get("max_agl_m")))
            sev = "warning" if d.get("outcome") in ("crashed", "ceiling", "telemetry_lost") else "info"
            agg.add("flybrain-racing-runs", "WINDOWS", path, day, f"{day}T00:00:00", msg, sev)
            n += 1
    for path in glob.glob(os.path.join(RACING_WORK, "train", "*", "flights.jsonl")):
        lines, off = S.read_new(path, state, full)
        new_state[path] = off
        day = dt.date.fromtimestamp(os.path.getmtime(path)).isoformat()
        for ln in lines:
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if not isinstance(d, dict) or "task" not in d:
                continue
            msg = ("liftoff train flight arm=%s round=%s task=%s profile=%s student_eval=%s outcome=%s success=%s "
                   "seconds=%s max_speed=%s max_tilt=%s min_agl=%s" % (
                       d.get("arm"), d.get("round"), d.get("task"), d.get("profile"), d.get("student_eval"),
                       d.get("outcome", d.get("error")), d.get("success"), d.get("seconds"),
                       d.get("max_speed_truth_ms"), d.get("max_tilt_truth_deg"), d.get("min_agl_truth_m")))
            sev = "warning" if d.get("outcome") in ("crashed", "ceiling", "arm_failed", "takeoff_failed") or "error" in d else "info"
            agg.add("flybrain-racing-train", "WINDOWS", path, day, f"{day}T00:00:00", msg, sev)
            n += 1
    for path in glob.glob(os.path.join(RACING_WORK, "train", "*", "events.jsonl")):
        lines, off = S.read_new(path, state, full)
        new_state[path] = off
        for ln in lines:
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if not isinstance(d, dict) or d.get("kind") not in ("collected", "eval", "start", "end", "paused_after_failures",
                                                                  "waiting", "focus_liftoff", "error"):
                continue
            day = str(d.get("t", ""))[:10] or dt.date.today().isoformat()
            msg = "liftoff train event %s arm=%s round=%s summary=%s" % (
                d.get("kind"), d.get("arm"), d.get("round"), json.dumps(d.get("summary"), sort_keys=True)[:400])
            sev = "warning" if d.get("kind") in ("paused_after_failures", "error") else "info"
            agg.add("flybrain-racing-train", "WINDOWS", path, day, str(d.get("t", day + "T00:00:00")), msg, sev)
            n += 1
    path = os.path.join(RACING_WORK, "obs_activity.log")
    if os.path.exists(path):
        lines, off = S.read_new(path, state, full)
        new_state[path] = off
        day = dt.date.fromtimestamp(os.path.getmtime(path)).isoformat()
        for ln in lines:
            m = re.match(r"^(\d\d:\d\d:\d\d) -> (\w+)", ln)
            if m:
                agg.add("flybrain-racing-obs", "WINDOWS", path, day, f"{day}T{m.group(1)}", f"obs scene -> {m.group(2)}", "info")
                n += 1
    return n


def extra(backfill=False):
    if backfill:
        return [("backfill-rotated", ship_backfill)]
    out = [
        ("wsl-journal", ship_journal),
        ("k3s-pods", ship_k3s_pods),
        ("k3s-events", ship_k3s_events),
        ("claude-tool-errors", ship_claude_errors),
        ("flybrain-run-log", ship_flybrain_runlogs),
        ("loci-audit", ship_glob("loci-audit", "WSL", WSLHOME + r"\.loci\audit\**\*.jsonl", jsonl=True)),
        ("loci-instrumentation", ship_glob("loci-instrumentation", "WSL", WSLHOME + r"\.loci\instrumentation\*.jsonl", jsonl=True)),
        ("cloudwatch", ship_cloudwatch),
        ("flybrain-dguard", ship_glob("flybrain-dguard", "WSL", WSLHOME + r"\.local\state\flybrain\*.log")),
    ]
    for name, p, health in LEDGERS:
        out.append((name, lambda a, s, ns, f, name=name, p=p, health=health: ship_litellm_ledger(a, s, ns, name, p, health, f)))
    for name, p in LITELLM_LOGS:
        out.append((name, lambda a, s, ns, f, name=name, p=p: ship_litellm(a, s, ns, name, p, f)))
    for log, src, lvl in (("Microsoft-Windows-WindowsUpdateClient/Operational", "win-windowsupdate", 3),
                          ("Microsoft-Windows-NTFS/Operational", "win-ntfs", 3)):
        out.append((src, lambda a, s, ns, f, log=log, src=src, lvl=lvl: S.ship_eventlog(a, s, ns, log, src, lvl)))
    out.append(("win-security-4625", lambda a, s, ns, f: S.ship_eventlog(a, s, ns, "Security", "win-security-failed-logon", ids=4625)))
    out.append(("win-liftoff-bepinex", lambda a, s, ns, f: ship_liftoff_log(a, s, ns, "win-liftoff-bepinex", LIFTOFF_BEPINEX, f, "bepinex")))
    out.append(("win-liftoff-player", lambda a, s, ns, f: ship_liftoff_log(a, s, ns, "win-liftoff-player", LIFTOFF_PLAYER, f, "player")))
    out.append(("flybrain-racing-runs", ship_liftoff_runs))

    # Copilot CLI session events — events.jsonl files in session-state dirs
    cop_sessions = os.path.join(S.HOME, ".copilot", "session-state")
    if os.path.isdir(cop_sessions):
        for session_id in sorted(os.listdir(cop_sessions)):
            events_path = os.path.join(cop_sessions, session_id, "events.jsonl")
            if not os.path.isfile(events_path):
                continue
            source_name = f"copilot-session:{session_id[:16]}"
            out.append((source_name, lambda a, s, ns, f, p=events_path, sn=source_name: S.ship_textfile(a, s, ns, sn, "WINDOWS", p, f, jsonl=True)))

    # Claude Code project session events — agent conversation JSONL files
    claude_projects = os.path.join(S.HOME, ".claude", "projects")
    if os.path.isdir(claude_projects):
        for project in sorted(os.listdir(claude_projects)):
            project_dir = os.path.join(claude_projects, project)
            if not os.path.isdir(project_dir):
                continue
            for fn in sorted(os.listdir(project_dir)):
                if fn.endswith(".jsonl"):
                    p = os.path.join(project_dir, fn)
                    source_name = f"claude-session:{fn[:20]}"
                    out.append((source_name, lambda a, s, ns, f, p=p, sn=source_name: S.ship_textfile(a, s, ns, sn, "WINDOWS", p, f, jsonl=True)))

    return out
