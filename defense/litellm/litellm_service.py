"""Supervisor for the LiteLLM proxy (run by the LiteLLM scheduled task).

- Loads secrets from the env file into the CHILD environment only.
- Binds loopback only; nothing here opens a listener beyond 127.0.0.1.
- Logs to logs/litellm.log (rotated at start when large).
- Restarts the proxy when it exits, or when /health/liveliness stops answering
  (a listening-but-wedged proxy looks "up" to the OS).
"""
import datetime
import os
import subprocess
import sys
import time
import urllib.request

BASE = r"F:\services\litellm"
CHECKOUT = r"F:\services\litellm\control"  # pinned control worktree @ df9e9acf; dev runs the working checkout
CONFIG = r"C:\Users\rjmendez-admin\AppData\Local\litellm\config.yaml"
ENV_FILE = r"C:\Users\rjmendez-admin\AppData\Local\litellm\.env"
PY = os.path.join(BASE, "venv", "Scripts", "python.exe")
HOST = os.environ.get("LITELLM_BIND_HOST", "127.0.0.1")
PORT = int(os.environ.get("LITELLM_BIND_PORT", "4000"))
LOG = os.path.join(BASE, "logs", "litellm.log")
GRACE_S = 180        # startup allowance before health checks count
PROBE_EVERY_S = 30
PROBE_TIMEOUT_S = 10
MAX_FAILS = 4        # consecutive failed probes before a restart
ROTATE_BYTES = 20 * 1024 * 1024


def log(msg):
    sys.stdout.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} [supervisor] {msg}\n")
    sys.stdout.flush()


def load_env():
    env = os.environ.copy()
    with open(ENV_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["LOCI_PHERO_DEPOSIT_DISABLED"] = "1"
    # 2026-10-18 migration to the published loci_litellm package: see F:\work\design\tripwire-governor-v2\prod_migrate.ps1
    env["LOCI_EMBED_API_BASE"] = "http://127.0.0.1:11435/v1"
    env["LOCI_EMBED_MODEL"] = "nomic-embed-text"
    env["LOCI_GATE_TOOLS_EXCLUDE_TAGS"] = "local"
    env["LOCI_LOCAL_PROVIDERS"] = "ollama"
    env["LOCI_CACHE_REGIME"] = "native-cc-20261018"
    env["LOCI_GOVERNOR"] = "enforce"   # 2026-10-29: enforce ON (envelope F:\services\litellm\governor\envelope.toml, pinned). Revert to "observe" to disable.
    env["LOCI_GOVERNOR_DIR"] = r"F:\services\litellm\governor"
    env["LOCI_ROUTER_SHADOW"] = "1"
    env["LOCI_ROUTER_LEARN_DIR"] = r"F:\services\litellm\router"
    env["LOCI_ROUTER_CANARY_SHARE"] = "0"
    env["PYTHONPATH"] = os.pathsep.join([
        r"F:\services\litellm\release\litellm-loci",   # published loci_litellm first: wins over the checkout copy
        CHECKOUT,
        os.path.join(CHECKOUT, "enterprise"),
        os.path.join(CHECKOUT, "litellm-proxy-extras"),
    ])
    return env


def healthy():
    try:
        with urllib.request.urlopen(f"http://{HOST}:{PORT}/health/liveliness", timeout=PROBE_TIMEOUT_S) as r:
            return r.status == 200
    except Exception:
        return False


def run_once():
    env = load_env()
    cmd = [PY, "-c", "from litellm import run_server; run_server()",
           "--config", CONFIG, "--host", HOST, "--port", str(PORT)]
    log(f"starting: {' '.join(cmd[:2])} ... --host {HOST} --port {PORT}")
    proc = subprocess.Popen(cmd, cwd=CHECKOUT, env=env,
                            stdout=sys.stdout, stderr=subprocess.STDOUT)
    started = time.time()
    fails = 0
    while True:
        try:
            rc = proc.wait(timeout=PROBE_EVERY_S)
            log(f"proxy exited rc={rc}")
            return
        except subprocess.TimeoutExpired:
            pass
        if time.time() - started < GRACE_S:
            continue
        if healthy():
            fails = 0
            continue
        fails += 1
        log(f"health probe failed ({fails}/{MAX_FAILS})")
        if fails >= MAX_FAILS:
            log("proxy unresponsive: killing for restart")
            proc.kill()
            proc.wait()
            return


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    if os.path.exists(LOG) and os.path.getsize(LOG) > ROTATE_BYTES:
        os.replace(LOG, LOG + ".1")
    sys.stdout = open(LOG, "a", encoding="utf-8", buffering=1)
    sys.stderr = sys.stdout
    backoff = 5
    while True:
        t0 = time.time()
        try:
            run_once()
        except Exception as exc:  # keep the supervisor alive; surface loudly
            log(f"supervisor error: {exc!r}")
        # a run that lived >5 min resets the backoff; fast crash loops back off to 5 min
        backoff = 5 if time.time() - t0 > 300 else min(backoff * 2, 300)
        log(f"restarting in {backoff}s")
        time.sleep(backoff)


if __name__ == "__main__":
    main()
