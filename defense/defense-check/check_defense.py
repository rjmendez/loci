"""Defense-in-depth status check for the Hermes -> LiteLLM -> Loci stack. Run on a schedule.

Each check compares the RUNNING state with the code and config on disk, and says which is missing:
  - a layer whose code changed after its process started is NOT live (the process has not loaded it)
  - a layer wired in config but not in code (or the reverse) is reported
  - uncommitted changes in a repo that holds live code are reported (they will not survive a rebuild)

Writes status.json next to this script and appends one line per run to status.log.
Exit 0 when every check passes, 1 when any check fails. Stdlib only.
"""
import datetime
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
LITELLM_CFG = os.path.join(HOME, "AppData", "Local", "litellm", "config.yaml")
LITELLM_ENV = os.path.join(HOME, "AppData", "Local", "litellm", ".env")
HERMES_CFG = os.path.join(HOME, "AppData", "Local", "hermes", "config.yaml")
HERMES_ALLOW = os.path.join(HOME, "AppData", "Local", "hermes", "shell-hooks-allowlist.json")
GUARD = os.path.join(HOME, "AppData", "Local", "hermes", "hooks", "guard_tool.py")
PLUGIN_REPO = r"F:\services\litellm\release\litellm-loci"
PLUGIN_POLICY = os.path.join(PLUGIN_REPO, "loci_litellm", "caller_policy.py")
LOCI_REPO = os.path.join(HOME, "loci-deploy")
LOCI_SERVER = os.path.join(LOCI_REPO, "mcp", "server.py")


def _ps(cmd: str) -> str:
    out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd],
                         capture_output=True, text=True, timeout=60)
    return out.stdout.strip()


def _mtime(path: str) -> float:
    return os.path.getmtime(path)


def _start_of(pid_expr: str) -> float:
    txt = _ps(f"$p=Get-CimInstance Win32_Process -Filter \"ProcessId=$({pid_expr})\"; "
              f"if($p){{ [int64]([DateTimeOffset]$p.CreationDate).ToUnixTimeSeconds() }} else {{ 0 }}")
    try:
        return float(txt or 0)
    except ValueError:
        return 0.0


def _listener_start(port: int) -> float:
    """Start time of the process that LISTENS on the port (the live one; wrappers and orphans do not)."""
    return _start_of(f"(Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
                     f"Select-Object -First 1).OwningProcess")


def _newest_match_start(match: str) -> float:
    """Start time of the NEWEST process whose command line matches (the live one when a stale copy exists)."""
    return _start_of(f"(Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -match '{match}' }} | "
                     f"Sort-Object CreationDate -Descending | Select-Object -First 1).ProcessId")


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=60).stdout.strip()


def check_litellm_proxy(results: dict) -> None:
    try:
        urllib.request.urlopen("http://127.0.0.1:4000/health/liveliness", timeout=5)
        results["litellm_healthy"] = True
    except Exception as exc:
        results["litellm_healthy"] = False
        results["litellm_healthy_detail"] = repr(exc)[:120]
    results["litellm_policy_in_config"] = "caller_policy.loci_caller_policy" in open(LITELLM_CFG, encoding="utf-8").read()
    started = _listener_start(4000)
    results["litellm_policy_loaded"] = bool(started) and started >= _mtime(PLUGIN_POLICY)
    results["litellm_policy_loaded_detail"] = (
        "process started after caller_policy.py last changed" if results["litellm_policy_loaded"]
        else "NOT LOADED: caller_policy.py changed after the proxy started; restart LiteLLM_Proxy")


def check_hermes(results: dict) -> None:
    cfg = open(HERMES_CFG, encoding="utf-8").read()
    results["hermes_guard_in_config"] = "guard_tool.py" in cfg and "fail_closed: true" in cfg
    allow = open(HERMES_ALLOW, encoding="utf-8").read() if os.path.exists(HERMES_ALLOW) else ""
    results["hermes_guard_consented"] = "guard_tool.py" in allow
    results["hermes_guard_file"] = os.path.exists(GUARD)
    results["hermes_identity_header_present"] = "x-loci-client:" in cfg
    # identity must reach the proxy: the LiteLLM-route providers entry carries the headers (merged before the client)
    results["hermes_identity_provider_entry"] = "litellm-hermes:" in cfg and "extra_headers:" in cfg.split("litellm-hermes:")[-1][:400]
    filt = os.path.join(HOME, "AppData", "Local", "hermes", "plugins", "loci-rag", "__init__.py")
    results["hermes_selfecho_filter"] = os.path.exists(filt) and "def drop_self_echo" in open(filt, encoding="utf-8").read()


def check_loci(results: dict) -> None:
    started = _newest_match_start("loci_launch\\.py")
    results["loci_server_running"] = bool(started)
    results["loci_server_has_current_code"] = bool(started) and started >= _mtime(LOCI_SERVER)
    results["loci_server_detail"] = (
        "running code is current" if results["loci_server_has_current_code"]
        else "NOT DEPLOYED: server.py changed after the Loci server started; restart via Loci-Watchdog")


def check_repos(results: dict) -> None:
    results["plugin_repo_dirty"] = bool(_git(PLUGIN_REPO, "status", "--porcelain"))
    results["loci_repo_dirty"] = bool(_git(LOCI_REPO, "status", "--porcelain"))
    results["plugin_repo_branch"] = _git(PLUGIN_REPO, "branch", "--show-current")


def main() -> int:
    results: dict = {"checked_at": datetime.datetime.now().isoformat(timespec="seconds")}
    for check in (check_litellm_proxy, check_hermes, check_loci, check_repos):
        try:
            check(results)
        except Exception as exc:
            results[check.__name__ + "_error"] = repr(exc)[:200]
    bad = []
    expect_true = ["litellm_healthy", "litellm_policy_in_config", "litellm_policy_loaded", "hermes_guard_in_config",
                   "hermes_guard_consented", "hermes_guard_file", "hermes_identity_provider_entry",
                   "hermes_selfecho_filter", "loci_server_running",
                   "loci_server_has_current_code"]
    for key in expect_true:
        if not results.get(key):
            bad.append(key)
    for key in ("plugin_repo_dirty", "loci_repo_dirty"):
        if results.get(key):
            bad.append(key)
    results["failed"] = bad
    results["ok"] = not bad and not any(k.endswith("_error") for k in results)
    with open(os.path.join(HERE, "status.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    with open(os.path.join(HERE, "status.log"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": results["checked_at"], "ok": results["ok"], "failed": bad}) + "\n")
    print(json.dumps({"ok": results["ok"], "failed": bad}))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
