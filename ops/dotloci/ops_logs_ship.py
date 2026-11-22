#!/usr/bin/env python3
"""Ship ops logs (Windows + WSL) into Qdrant collection `ops_logs` for RAG.

Read-only on all log sources. Idempotent: point ids are uuid5(source|day|signature),
so re-runs upsert. Repeated lines collapse to one point per (source, day, signature)
with a count. Secrets are scrubbed before anything is embedded or stored.

Query with: rag_context_search(query, collections=["ops_logs"]).
Usage: python ops_logs_ship.py [--dry-run] [--only NAME] [--full]
"""
import argparse, collections, datetime as dt, json, os, re, subprocess, sys, uuid
import requests

HOME = os.path.expanduser("~")
QDRANT = os.environ.get("QDRANT_URL", "http://127.0.0.1:30633")
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11435")
COLL = "ops_logs"
AGENT_COLL = "ops_logs_agent"   # Claude/agent tool errors: kept apart so they do not crowd ops results


def coll_for(source):
    return AGENT_COLL if source.startswith("claude-") else COLL

PSEXE = "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"   # pwsh is not on PATH under Task Scheduler
EMBED_MODEL = "nomic-embed-text"
STATE = os.path.join(HOME, ".loci", "ops_logs_state.json")
WSL_DEFAULT_DISTRO = "Ubuntu"          # the pre-2026-10-14 distro; the last resort when nothing better is found
WSL_USER = os.environ.get("LOCI_WSL_USER", "rjmendez")
WSL = r"\\wsl.localhost" + "\\" + WSL_DEFAULT_DISTRO
WSLHOME = WSL + "\\home\\" + WSL_USER
WSL_DISTRO, WSL_DISTRO_HOW, WSL_CANDIDATES = WSL_DEFAULT_DISTRO, "default", []
SHIP_HEALTH = os.path.join(HOME, ".loci", "ops_logs_ship_health.json")   # per-source status of the last run (the self-test reads it)


class SourceUnavailable(Exception):
    """A source that cannot be collected on this machine right now (file or tool absent). Reported loudly, never a crash."""


def decode_wsl_list(raw):
    """Distro names from `wsl.exe -l -q [--running]` output bytes. wsl.exe writes UTF-16LE (usually with a BOM) when
    piped; newer builds can write UTF-8. Both are accepted. Blank lines, the ' (Default)' tag and any line that is
    not a name (a localised 'no running distributions' sentence has spaces) are dropped."""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        txt = raw.decode("utf-16", "replace")
    elif b"\x00" in raw:
        txt = raw.decode("utf-16-le", "replace")
    else:
        txt = raw.decode("utf-8", "replace")
    names = []
    for ln in txt.replace("﻿", "").splitlines():
        ln = re.sub(r"\s*\(Default\)\s*$", "", ln.replace("\x00", "").strip())
        if ln and not re.search(r"\s", ln):
            names.append(ln)
    return names


def running_wsl_distros(run=None):
    run = run or subprocess.run
    try:
        r = run(["wsl.exe", "-l", "-q", "--running"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    return decode_wsl_list(r.stdout) if r.returncode == 0 else []


def detect_wsl_distro(env=None, run=None, exists=os.path.exists):
    """Pick the WSL distro whose files the shipper reads. Order: env LOCI_WSL_DISTRO (trusted as given); else the first
    RUNNING distro that has /var/log/syslog (probing a stopped distro's UNC path would start it, so only running ones
    are probed); else the old default. Returns {distro, how, candidates}."""
    env = os.environ if env is None else env
    forced = (env.get("LOCI_WSL_DISTRO") or "").strip()
    if forced:
        return {"distro": forced, "how": "env", "candidates": []}
    cands = running_wsl_distros(run)   # None -> subprocess.run, looked up at call time
    for d in cands:
        if exists(r"\\wsl.localhost" + "\\" + d + r"\var\log\syslog"):
            return {"distro": d, "how": "running+syslog", "candidates": cands}
    return {"distro": WSL_DEFAULT_DISTRO, "how": "default", "candidates": cands}


_wsl_resolved = False


def resolve_wsl(force=False, **kw):
    """Detect once per process and point WSL / WSLHOME (read by ops_logs_ext at call time) at the chosen distro."""
    global WSL, WSLHOME, WSL_DISTRO, WSL_DISTRO_HOW, WSL_CANDIDATES, _wsl_resolved
    if _wsl_resolved and not force:
        return {"distro": WSL_DISTRO, "how": WSL_DISTRO_HOW, "candidates": WSL_CANDIDATES}
    d = detect_wsl_distro(**kw)
    WSL_DISTRO, WSL_DISTRO_HOW, WSL_CANDIDATES = d["distro"], d["how"], d["candidates"]
    WSL = r"\\wsl.localhost" + "\\" + WSL_DISTRO
    WSLHOME = WSL + "\\home\\" + WSL_USER
    _wsl_resolved = True
    return d


def wsl_argv(user=None):
    """wsl.exe argv that runs a script on the chosen distro. With nothing detected, no -d: wsl.exe's own default
    distro, which is what the shipper has always used for its in-distro commands."""
    base = ["wsl.exe"] + (["-d", WSL_DISTRO] if WSL_DISTRO_HOW in ("env", "running+syslog") else [])
    return base + (["-u", user] if user else []) + ["-e", "bash", "-s"]


def require_wsl_command(cmd, run=None):
    """Raise SourceUnavailable (with the distro named) unless `cmd` exists on the chosen distro."""
    run = run or subprocess.run
    try:
        r = run(wsl_argv(), input=f"command -v {cmd} >/dev/null 2>&1\n".encode(), capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise SourceUnavailable(f"wsl.exe unusable: {type(e).__name__}: {e}")
    if r.returncode != 0:
        raise SourceUnavailable(f"`{cmd}` not found on WSL distro {WSL_DISTRO} ({WSL_DISTRO_HOW})")


def ship_wsl_file(agg, state, new_state, source, host, path, full):
    """ship_textfile for a file that may legitimately not exist on this distro: say so instead of silently shipping 0."""
    try:
        exists = os.path.exists(path)
    except OSError:
        exists = False
    if not exists:
        raise SourceUnavailable(f"file not found: {path}")
    try:
        return ship_textfile(agg, state, new_state, source, host, path, full)
    except PermissionError:   # 0640 syslog:adm and the distro user is not in adm: read it as root via wsl.exe
        return ship_textfile(agg, state, new_state, source, host, path, full, via_wsl=True)
NS = uuid.UUID("6f1c1d3e-0000-4000-8000-0b5109000001")
MAX_FIRST_RUN_BYTES = 40_000_000   # per file, newest bytes only
SEV = re.compile(r"\b(CRITICAL|FATAL|ERROR|ERR|WARN(?:ING)?|INFO|DEBUG)\b", re.I)
TS = re.compile(r"^\[?(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})")
SECRET = [
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{12,}"), r"\1***"),
    (re.compile(r"(?i)((?:api[_-]?key|token|secret|password|passwd|authorization)[\"'\s:=]+)[^\s\"',;]{6,}"), r"\1***"),
    (re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b"), "***AWSKEY***"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "***GHTOKEN***"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), "***SKKEY***"),
]
NORM = [
    (re.compile(r"[0-9a-f]{8,}", re.I), "<hex>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "<ts>"),
    (re.compile(r"\d+"), "<n>"),
]


def scrub(s):
    for rx, rep in SECRET:
        s = rx.sub(rep, s)
    return s


def signature(line):
    s = line
    for rx, rep in NORM:
        s = rx.sub(rep, s)
    return s[:300]


GLOG = re.compile(r"\b([EWI])\d{4} \d\d:\d\d:\d\d")
BAD = re.compile(r"(?i)\b(failed|failure|panic|oom|out of memory|denied|refused|timed? ?out|traceback|exception|unreachable|crash|segfault)\b|\w+Error\b|HTTP/1\.1\" [45]\d\d")


def severity(line):
    g = GLOG.search(line)
    if g and g.group(1) != "I":
        return "error" if g.group(1) == "E" else "warning"
    m = SEV.search(line)
    if not m:
        return "warning" if BAD.search(line) else "info"
    v = m.group(1).upper()
    return {"ERR": "error", "FATAL": "critical", "CRITICAL": "critical", "ERROR": "error",
            "WARN": "warning", "WARNING": "warning", "INFO": "info", "DEBUG": "debug"}[v]


# Known-benign lines that merely contain alarming words (e.g. "crash"); extend with OPS_NOISE_REGEX.
NOISE_RX = re.compile(os.environ.get("OPS_NOISE_REGEX") or r"apport\.service - automatic crash report generation was skipped", re.I)
FLOOR = {"wsl-auth", "wsl-syslog", "wsl-kern", "win-copilot", "wsl-hermes-agent", "wsl-hermes-gateway"}  # info/debug dropped; errors+warnings kept


class Agg:
    """Collapse lines to (source, day, signature) -> count, first/last, samples."""
    def __init__(self):
        self.d = {}

    def add(self, source, host, path, day, tstr, line, sev=None):
        line = scrub(line.rstrip())[:1500]
        if not line.strip() or "rt_tgrep" in line or NOISE_RX.search(line):
            return
        if any(source == f or source.startswith(f + "-") for f in FLOOR) and (sev or severity(line)) in ("info", "debug"):
            return
        sig = signature(line)
        k = (source, day, sig)
        e = self.d.get(k)
        if e is None:
            self.d[k] = {"source": source, "host": host, "path": path, "day": day,
                         "severity": sev or severity(line), "count": 1,
                         "first": tstr, "last": tstr, "sample": line, "sample_last": line}
        else:
            e["count"] += 1
            e["last"] = tstr
            e["sample_last"] = line


def start_offset(size, off, full):
    if off > size:                       # rotated/truncated
        off = 0
    if off == 0 and size > MAX_FIRST_RUN_BYTES:
        off = size - MAX_FIRST_RUN_BYTES
    if full:
        off = max(0, size - MAX_FIRST_RUN_BYTES)
    return off


def complete_lines(data, off):
    """(lines, new offset) from bytes read at `off`: only whole lines are consumed."""
    cut = data.rfind(b"\n")
    if cut < 0:
        return [], off
    data = data[:cut + 1]
    return data.decode("utf-8", "replace").splitlines(), off + len(data)


def read_new(path, state, full):
    try:
        size = os.path.getsize(path)
    except OSError:
        return [], 0
    off = start_offset(size, state.get(path, 0), full)
    with open(path, "rb") as f:
        f.seek(off)
        data = f.read()
    return complete_lines(data, off)


def read_new_via_wsl(path, state, full, run=None):
    """read_new for a file on a WSL distro share that Windows cannot open (PermissionError: /var/log/syslog
    is syslog:adm 0640 and the distro user is not in adm). Reads it as root through wsl.exe instead. Returns
    (lines, new_off, mtime). Read-only: stat and tail."""
    run = run or subprocess.run
    lin = path[len(WSL):].replace("\\", "/")
    q = "'" + lin.replace("'", "'\\''") + "'"
    r = run(wsl_argv("root"), input=f"stat -c '%s %Y' {q}\n".encode(), capture_output=True, timeout=60)
    try:
        size, mtime = (int(x) for x in r.stdout.decode().split())
    except ValueError:
        raise SourceUnavailable(f"cannot stat {lin} on WSL distro {WSL_DISTRO}: {r.stderr.decode('utf-8', 'replace').strip()[:150]}")
    off = start_offset(size, state.get(path, 0), full)
    if off >= size:
        return [], off, mtime
    r = run(wsl_argv("root"), input=f"tail -c +{off + 1} {q} | head -c {MAX_FIRST_RUN_BYTES}\n".encode(),
            capture_output=True, timeout=300)
    lines, new_off = complete_lines(r.stdout, off)
    return lines, new_off, mtime


def ship_textfile(agg, state, new_state, source, host, path, full, jsonl=False, via_wsl=False):
    if via_wsl:
        lines, new_off, mtime = read_new_via_wsl(path, state, full)
        mday = dt.date.fromtimestamp(mtime).isoformat()
    else:
        lines, new_off = read_new(path, state, full)
        mday = dt.date.fromtimestamp(os.path.getmtime(path)).isoformat() if os.path.exists(path) else "unknown"
    new_state[path] = new_off
    cur_day, cur_t = mday, ""
    for ln in lines:
        m = TS.match(ln)
        if m:
            cur_day, cur_t = m.group(1), m.group(2)
        elif jsonl:
            m2 = re.search(r'"(?:ts|timestamp)"\s*:\s*"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})', ln)
            if m2:
                cur_day, cur_t = m2.group(1), m2.group(2)
        agg.add(source, host, path, cur_day, f"{cur_day}T{cur_t}" if cur_t else cur_day, ln)
    return len(lines)


GIN = re.compile(r'^\[GIN\] (\d{4})/(\d\d)/(\d\d) - (\d\d:\d\d:\d\d) \|\s*(\d{3}) \|\s*(?:(\d+)m)?([\d.]+)([^\d\s|.]*s) \|\s*(\S+) \| (\w+)\s+"([^"]*)"')
SLOG = re.compile(r'^time=(\d{4}-\d\d-\d\d)T(\d\d:\d\d:\d\d)\S* level=(\w+) source=\S+ msg=(.*)$')
# lifecycle lines worth keeping at INFO: runner starts/stops, memory placement decisions, load failures
OLLAMA_INFO = re.compile(r'starting llama-server|loading model via|predicted to exceed|fits alongside|Load failed|llama-server terminated|'
                         r'llama-server started|loaded runners|selecting (?:single )?GPU|evict|unload', re.I)
GIN_SLOW_S = 20.0
CTX_WORD = {2048: "two_k", 4096: "four_k", 8192: "eight_k", 16384: "sixteen_k", 32768: "thirtytwo_k", 65536: "sixtyfour_k",
            131072: "onetwentyeight_k", 262144: "twofiftysix_k"}


def ship_ollama(agg, state, new_state, source, host, path, full):
    """Ollama server log. WHITELIST only: its slog lines and GIN access lines. Everything else in the file is dropped, because the
    verbose llama.cpp output echoes prompt text (`<|im_start|>user ...`) and must never reach the index. Healthy 200s are dropped;
    kept are 4xx/5xx, requests slower than GIN_SLOW_S, WARN/ERROR slog lines and model lifecycle lines (with their -c / -np args)."""
    lines, new_off = read_new(path, state, full)
    new_state[path] = new_off
    for ln in lines:
        g = GIN.match(ln)
        if g:
            y, mo, d, hms, status, mins, secs, unit, ip, method, ep = g.groups()
            status = int(status)
            lat = float(secs) * {"s": 1.0, "ms": 1e-3}.get(unit, 1e-6) + 60.0 * int(mins or 0)
            if status < 400 and lat < GIN_SLOW_S:
                continue
            sev = "error" if status >= 500 else "warning"
            what = f"slow request {lat:.1f}s" if status < 400 else f"http {status}"
            agg.add(source, host, path, f"{y}-{mo}-{d}", f"{y}-{mo}-{d}T{hms}", f"ollama {what} {method} {ep} from {ip} ({lat:.1f}s)", sev)
            continue
        s = SLOG.match(ln)
        if s:
            day, hms, level, msg = s.groups()
            if level in ("WARN", "WARNING"):
                sev = "warning"
            elif level == "ERROR":
                sev = "error"
            elif OLLAMA_INFO.search(msg) and "server config" not in msg:
                sev = "info"
            else:
                continue
            if "starting llama-server" in msg:   # compact it: keep ctx and parallel slots, drop the long argv; spell the class out so
                c = re.search(r"-c (\d+) -np (\d+)", msg)   # the digit-normalised signature still tells 16k from 64k from 256k
                if c:
                    n = int(c.group(1))
                    msg = f'"starting llama-server" ctx={n} np={c.group(2)} ctx_class={CTX_WORD.get(n, "other")} total_ctx={n * int(c.group(2))}'
            agg.add(source, host, path, day, f"{day}T{hms}", f"ollama {level} {msg}", sev)
            continue
        if ln.startswith("Error:"):   # e.g. "listen tcp 0.0.0.0:11434: bind: Only one usage ..." from a duplicate start
            m = re.search(r"\d{4}-\d\d-\d\d", ln)
            day = m.group(0) if m else dt.date.fromtimestamp(os.path.getmtime(path)).isoformat()
            agg.add(source, host, path, day, day, "ollama " + ln, "error")
    return len(lines)


def ship_eventlog(agg, state, new_state, logname, source, max_level=3, limit=3000, ids=None):
    key = f"evtlog:{logname}"
    since = state.get(key)
    ps = ("$ErrorActionPreference='SilentlyContinue';"
          f"$f=@{{LogName='{logname}';" + (f"Id={ids}" if ids else f"Level=1..{max_level}") + "};"
          + (f"$f.StartTime=[datetime]'{since}';" if since else "$f.StartTime=(Get-Date).AddDays(-30);")
          + f"Get-WinEvent -FilterHashtable $f -MaxEvents {limit} | "
          "Select-Object @{n='t';e={$_.TimeCreated.ToUniversalTime().ToString('o')}},Id,ProviderName,LevelDisplayName,"
          "@{n='m';e={($_.Message -replace '\\s+',' ')}} | ConvertTo-Json -Compress")
    r = subprocess.run([PSEXE, "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=300)
    out = r.stdout.strip()
    if not out:
        new_state[key] = since or dt.datetime.utcnow().isoformat()
        return 0
    rows = json.loads(out)
    if isinstance(rows, dict):
        rows = [rows]
    newest = since or ""
    for ev in rows:
        t = ev["t"]
        newest = max(newest, t)
        sev = {"Critical": "critical", "Error": "error", "Warning": "warning"}.get(ev["LevelDisplayName"], "info")
        if ids:   # audit-failure style events carry Information level; still alert-worthy
            sev = "warning"
        line = f'{ev["ProviderName"]} id={ev["Id"]} {ev["LevelDisplayName"]}: {ev["m"]}'
        agg.add(source, "WINDOWS", f"eventlog:{logname}", t[:10], t, line, sev)
    new_state[key] = newest[:19] if newest else (since or "")
    return len(rows)


def collectors():
    resolve_wsl()   # before ops_logs_ext is imported: it reads WSLHOME at import time
    wl = os.path.join(HOME, ".loci", "offload_call_telemetry.jsonl")
    cop = os.path.join(HOME, ".copilot", "logs")
    out = []
    for n in ("System", "Application"):
        out.append((f"win-eventlog-{n.lower()}", lambda a, s, ns, f, n=n: ship_eventlog(a, s, ns, n, f"win-eventlog-{n.lower()}")))
    out.append(("win-powershell", lambda a, s, ns, f: ship_eventlog(a, s, ns, "Microsoft-Windows-PowerShell/Operational", "win-powershell")))
    out.append(("win-defender", lambda a, s, ns, f: ship_eventlog(a, s, ns, "Microsoft-Windows-Windows Defender/Operational", "win-defender")))
    out.append(("win-hyperv", lambda a, s, ns, f: ship_eventlog(a, s, ns, "Microsoft-Windows-Hyper-V-Compute-Admin", "win-hyperv", 4)))
    out.append(("win-loci-offload", lambda a, s, ns, f: ship_textfile(a, s, ns, "win-loci-offload", "WINDOWS", wl, f, jsonl=True)))
    for name, p in (("win-ollama-serve", os.path.join(HOME, "ollama_serve.log")),
                    ("win-ollama-applog", os.path.join(os.environ.get("LOCALAPPDATA", ""), "Ollama", "server.log"))):
        if os.path.exists(p):
            out.append((name, lambda a, s, ns, f, p=p, name=name: ship_ollama(a, s, ns, name, "WINDOWS", p, f)))
    if os.path.isdir(cop):
        for fn in sorted(os.listdir(cop)):
            if fn.endswith(".log"):
                p = os.path.join(cop, fn)
                out.append((f"win-copilot:{fn}", lambda a, s, ns, f, p=p: ship_textfile(a, s, ns, "win-copilot", "WINDOWS", p, f)))
    wsl = [("wsl-syslog", r"\var\log\syslog"), ("wsl-kern", r"\var\log\kern.log"), ("wsl-auth", r"\var\log\auth.log"),
           ("wsl-dama-sqs", r"\var\log\dama-sqs-consumer.log"), ("wsl-tmux-watchdog", r"\var\log\tmux-watchdog.log")]
    for name, rel in wsl:
        p = WSL + rel
        out.append((name, lambda a, s, ns, f, p=p, name=name: ship_wsl_file(a, s, ns, name, "WSL", p, f)))
    for fn in ("agent.log", "errors.log", "gateway.log", "mcp-stderr.log", "update.log", "gateway_faulthandler.log"):
        p = WSLHOME + r"\.hermes\logs" + "\\" + fn   # tool-audit.log skipped: records this tooling's own calls
        out.append((f"wsl-hermes-{fn}", lambda a, s, ns, f, p=p, fn=fn: ship_wsl_file(a, s, ns, f"wsl-hermes-{fn[:-4]}", "WSL", p, f)))
    sys.modules.setdefault("ops_logs_ship", sys.modules[__name__])
    import ops_logs_ext
    return ops_logs_ext.extra(True) if os.environ.get("OPS_BACKFILL") == "1" else out + ops_logs_ext.extra()


def embed(texts, timeout=300):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=timeout)
    r.raise_for_status()
    return r.json()["embeddings"]


def qheaders():
    key = os.environ.get("QDRANT_API_KEY")
    if not key:
        import tomllib
        with open(os.path.join(HOME, ".loci", "backends.toml"), "rb") as f:
            key = tomllib.load(f)["qdrant"].get("api_key")
    return {"api-key": key} if key else {}


def ensure_collection(h, coll=COLL):
    r = requests.get(f"{QDRANT}/collections/{coll}", headers=h, timeout=30)
    if r.status_code == 200:
        return
    r = requests.put(f"{QDRANT}/collections/{coll}", headers=h, timeout=30,
                     json={"vectors": {"size": 768, "distance": "Cosine"}})
    r.raise_for_status()
    for fld in ("source", "host", "severity", "day"):
        requests.put(f"{QDRANT}/collections/{coll}/index", headers=h, timeout=30,
                     json={"field_name": fld, "field_schema": "keyword"})


def upsert(items, coll, h, full):
    B = 64
    for i in range(0, len(items), B):
        batch = items[i:i + B]
        ids = [str(uuid.uuid5(NS, f'{x["source"]}|{x["day"]}|{signature(x["sample"])}')) for x in batch]
        old = {}
        if not full:   # merge with the day's earlier count so incremental runs accumulate
            rr = requests.post(f"{QDRANT}/collections/{coll}/points", headers=h, timeout=60,
                               json={"ids": ids, "with_payload": ["count", "first"], "with_vector": False})
            if rr.ok:
                old = {p["id"]: p["payload"] for p in rr.json()["result"]}
        for x, pid in zip(batch, ids):
            if pid in old:
                x["count"] += old[pid].get("count", 0)
                x["first"] = old[pid].get("first") or x["first"]
        texts = [f'[{x["source"]} {x["host"]} {x["day"]} {x["severity"]} x{x["count"]}] {x["sample"]}'[:1800] for x in batch]
        vecs = embed(texts)
        pts = [{"id": pid, "vector": v, "payload": {**x, "type": "ops_log", "text": t}}
               for x, pid, v, t in zip(batch, ids, vecs, texts)]
        r = requests.put(f"{QDRANT}/collections/{coll}/points?wait=true", headers=h, json={"points": pts}, timeout=120)
        r.raise_for_status()
    os.makedirs(os.path.dirname(STATE), exist_ok=True)


def write_ship_health(statuses, path=None, now=None):
    """Record which WSL distro was chosen and each source's outcome (atomic write). Hundreds of per-session sources
    (names with ':') are only listed when they are not ok; everything is counted in `totals`."""
    path = path or SHIP_HEALTH
    totals = collections.Counter(v["status"] for v in statuses.values())
    listed = {k: v for k, v in statuses.items() if ":" not in k or v["status"] != "ok"}
    doc = {"ts": (now or dt.datetime.now()).isoformat(timespec="seconds"),
           "wsl": {"distro": WSL_DISTRO, "how": WSL_DISTRO_HOW, "running": list(WSL_CANDIDATES)},
           "totals": dict(totals), "sources": listed}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    try:
        state = json.load(open(STATE))
    except Exception:
        state = {}
    new_state = dict(state)
    agg = Agg()
    statuses = {}
    cols = collectors()
    print(f"wsl distro: {WSL_DISTRO} ({WSL_DISTRO_HOW}; running: {WSL_CANDIDATES})", flush=True)
    for name, fn in cols:
        if a.only and a.only not in name:
            continue
        try:
            n = fn(agg, state, new_state, a.full)
            print(f"{name}: {n} new lines/events", flush=True)
            statuses[name] = {"status": "ok", "n": n}
        except SourceUnavailable as e:
            print(f"{name}: SKIPPED (unavailable) {e}", flush=True)
            statuses[name] = {"status": "unavailable", "detail": str(e)[:300]}
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
            statuses[name] = {"status": "failed", "detail": f"{type(e).__name__}: {e}"[:300]}
    skipped = sorted(k for k, v in statuses.items() if v["status"] == "unavailable")
    if skipped:
        print(f"UNAVAILABLE sources ({len(skipped)}): {skipped}", flush=True)
    items = list(agg.d.values())
    by = collections.Counter(i["source"] for i in items)
    print(f"{len(items)} signature points; per source: {dict(by)}")
    if not a.dry_run:
        write_ship_health(statuses)
    if a.dry_run:
        for i in sorted(items, key=lambda x: -x["count"])[:8]:
            print(i["count"], i["severity"], i["source"], i["sample"][:140])
        return
    h = qheaders()
    for coll in (COLL, AGENT_COLL):
        sub = [x for x in items if coll_for(x["source"]) == coll]
        if sub:
            ensure_collection(h, coll)
            upsert(sub, coll, h, a.full)
    try:   # merge: another run (scheduled vs manual) may have advanced other keys meanwhile
        cur = json.load(open(STATE))
    except Exception:
        cur = {}
    cur.update({k: v for k, v in new_state.items() if state.get(k) != v})
    json.dump(cur, open(STATE, "w"))
    print("done")


if __name__ == "__main__":
    sys.exit(main())
