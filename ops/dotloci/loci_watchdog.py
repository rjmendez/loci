"""Keep the live Loci server up. Run once a minute by the Loci-Watchdog scheduled task.

Loci has no supervisor: on 2026-10-19 16:20 its process was killed from outside and stayed down until someone noticed.
This probes http://127.0.0.1:8000/health and runs start_loci.ps1 when the server has been down long enough to be sure.

Rules (decide() below, pure so it is testable):
  * a healthy probe resets the failure count;
  * process GONE:  restart after DEAD_FAILS consecutive failed probes (2 minutes);
  * process ALIVE but not answering (hung, or still booting): restart only after HUNG_FAILS (6 minutes), because
    start_loci.ps1 kills whatever listens on :8000 and a slow boot must not be killed;
  * never two restarts less than COOLDOWN_S apart; at most MAX_RESTARTS_PER_HOUR, then it stops restarting and says so
    ("give_up") instead of looping on a server that cannot start.
Every decision other than "healthy" is logged to ~/.loci/loci_watchdog.log; state is ~/.loci/loci_watchdog_state.json.

LiteLLM proxy watchdog (2026-10-28): separately probes http://127.0.0.1:4000/health/liveliness and runs
Start-ScheduledTask LiteLLM_Proxy when it is unreachable.  The LiteLLM task's own RestartOnFailure handles
proxy crashes; this watchdog recovers the case where the supervisor process is killed externally (e.g. sleep/
hibernate) and the task scheduler does not detect the process exit.  Uses LITELLM_DEAD_FAILS consecutive
misses before acting to avoid false positives during a slow cold start.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

HEALTH_URL = "http://127.0.0.1:8000/health"
PROBE_TIMEOUT_S = 10
DEAD_FAILS = 2
HUNG_FAILS = 6
COOLDOWN_S = 300
MAX_RESTARTS_PER_HOUR = 4
LOCI_DIR = os.path.join(os.path.expanduser("~"), ".loci")
STATE_PATH = os.path.join(LOCI_DIR, "loci_watchdog_state.json")
LOG_PATH = os.path.join(LOCI_DIR, "loci_watchdog.log")
START_SCRIPT = os.path.join(LOCI_DIR, "start_loci.ps1")

# LiteLLM proxy watchdog constants
LITELLM_HEALTH_URL = "http://127.0.0.1:4000/health/liveliness"
LITELLM_STATE_PATH = os.path.join(LOCI_DIR, "litellm_watchdog_state.json")
LITELLM_TASK_NAME = "LiteLLM_Proxy"
LITELLM_DEAD_FAILS = 3   # 3 consecutive misses (~3 minutes) before restart; slow cold start can take 60-90 s
LITELLM_MAX_RESTARTS_PER_HOUR = 3
LITELLM_COOLDOWN_S = 300


def decide(state, healthy, process_alive, now):
    """-> (action, new_state). action is one of ok, wait_dead, wait_hung, cooldown, give_up, restart."""
    fails = 0 if healthy else int(state.get("fails", 0)) + 1
    restarts = [t for t in state.get("restarts", []) if now - t < 3600]
    new = {"fails": fails, "restarts": restarts}
    if healthy:
        return "ok", new
    need = HUNG_FAILS if process_alive else DEAD_FAILS
    if fails < need:
        return ("wait_hung" if process_alive else "wait_dead"), new
    if len(restarts) >= MAX_RESTARTS_PER_HOUR:
        return "give_up", new
    if restarts and now - max(restarts) < COOLDOWN_S:
        return "cooldown", new
    new["restarts"] = restarts + [now]
    new["fails"] = 0
    return "restart", new


def probe():
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=PROBE_TIMEOUT_S) as r:
            return r.status == 200
    except Exception:
        return False


def loci_process_alive():
    """True if a python process is running loci_launch.py. A query failure counts as alive: never restart on a guess."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
          "Where-Object { $_.CommandLine -match 'loci_launch\\.py' } | Measure-Object | Select-Object -ExpandProperty Count")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=30, check=True).stdout
        return int(out.strip()) > 0
    except Exception as exc:
        log(f"process query failed ({exc!r}); treating Loci as alive so nothing is restarted on a guess")
        return True


def log(msg):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def load_state():
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log(f"state file unreadable ({exc!r}); starting from empty state")
        return {}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)


def main():
    state = load_state()
    healthy = probe()
    alive = True if healthy else loci_process_alive()
    action, new = decide(state, healthy, alive, time.time())
    if action != "ok":
        log(f"health failed (consecutive {new['fails']}, process {'alive' if alive else 'GONE'}): {action}")
    if action == "restart":
        r = subprocess.run(["powershell", "-NoProfile", "-File", START_SCRIPT, "-NoWait"], capture_output=True, text=True, timeout=120)
        log(f"restart issued: start_loci.ps1 exit {r.returncode} {(r.stdout + r.stderr).strip()[:300]}")
    elif action == "give_up":
        log(f"GIVING UP: {MAX_RESTARTS_PER_HOUR} restarts in the last hour and Loci is still not answering; fix it by hand")
    save_state(new)
    return 1 if action == "give_up" else 0


def probe_litellm() -> bool:
    try:
        with urllib.request.urlopen(LITELLM_HEALTH_URL, timeout=PROBE_TIMEOUT_S) as r:
            return r.status == 200
    except Exception:
        return False


def load_litellm_state() -> dict:
    try:
        with open(LITELLM_STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log(f"litellm state file unreadable ({exc!r}); starting from empty state")
        return {}


def save_litellm_state(state: dict) -> None:
    tmp = LITELLM_STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, LITELLM_STATE_PATH)


def watch_litellm() -> None:
    """Probe the LiteLLM proxy and start the scheduled task if it is down long enough."""
    state = load_litellm_state()
    now = time.time()
    if probe_litellm():
        save_litellm_state({"fails": 0, "restarts": state.get("restarts", [])})
        return
    fails = int(state.get("fails", 0)) + 1
    restarts = [t for t in state.get("restarts", []) if now - t < 3600]
    new_state = {"fails": fails, "restarts": restarts}
    save_litellm_state(new_state)
    if fails < LITELLM_DEAD_FAILS:
        log(f"litellm proxy unreachable (fail {fails}/{LITELLM_DEAD_FAILS}); waiting")
        return
    if len(restarts) >= LITELLM_MAX_RESTARTS_PER_HOUR:
        log(f"litellm watchdog GIVING UP: {LITELLM_MAX_RESTARTS_PER_HOUR} restarts in the last hour; fix it by hand")
        return
    if restarts and now - max(restarts) < LITELLM_COOLDOWN_S:
        log(f"litellm proxy still down but in cooldown ({int(now - max(restarts))}s < {LITELLM_COOLDOWN_S}s)")
        return
    log(f"litellm proxy unreachable for {fails} consecutive probes; starting {LITELLM_TASK_NAME}")
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"Start-ScheduledTask -TaskName '{LITELLM_TASK_NAME}'"],
        capture_output=True, text=True, timeout=30,
    )
    log(f"Start-ScheduledTask {LITELLM_TASK_NAME}: exit {r.returncode} {(r.stdout + r.stderr).strip()[:200]}")
    new_state["fails"] = 0
    new_state["restarts"] = restarts + [now]
    save_litellm_state(new_state)


MIN_RUN_GAP_S = 45                      # a lone task runs every 60 s; a second task running this script must not double the cadence
RUN_LOCK = os.path.join(LOCI_DIR, "loci_watchdog.run.lock")
RUN_STAMP = os.path.join(LOCI_DIR, "loci_watchdog.run.stamp")


def claim_run(now=None) -> bool:
    """True if this run may proceed. 2026-11-05: Loci-Watchdog (logon) and Loci-Watchdog-Boot (boot) both ran this every
    minute, so probes were counted ~13 s apart, every grace period shrank, and the hourly restart budget drained. Take an
    exclusive lock (stale after 120 s) and refuse when the previous run started less than MIN_RUN_GAP_S ago."""
    now = time.time() if now is None else now
    try:
        fd = os.open(RUN_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            if now - os.path.getmtime(RUN_LOCK) < 120:
                return False
            os.remove(RUN_LOCK)
            fd = os.open(RUN_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            return False
    except OSError:
        return True                      # cannot lock (read-only dir, ...): run rather than silently never watch
    os.close(fd)
    try:
        if now - os.path.getmtime(RUN_STAMP) < MIN_RUN_GAP_S:
            os.remove(RUN_LOCK)
            return False
    except OSError:
        pass                             # no stamp yet
    try:
        with open(RUN_STAMP, "w") as f:
            f.write(str(now))
    except OSError:
        pass
    return True


def release_run() -> None:
    try:
        os.remove(RUN_LOCK)
    except OSError:
        pass


if __name__ == "__main__":
    if not claim_run():
        sys.exit(0)
    try:
        watch_litellm()
        rc = main()
    finally:
        release_run()
    sys.exit(rc)
