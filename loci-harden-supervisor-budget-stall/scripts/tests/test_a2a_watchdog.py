"""scripts/a2a_watchdog.py — a probe that could not run must not read as healthy.

The file's contract is 'silent on healthy; prints only on problems', so silence
is what the operator (and cron) reads as 'the portproxy is fine'. Both read
probes used to collapse an exception into the same None a successful read with
no match returns, producing two opposite wrong answers:

  * get_wsl_ip failing skipped BOTH drift branches -> total silence while the
    A2A server was unreachable over Tailscale.
  * get_portproxy_wsl_ip failing rendered as 'No portproxy rule found', and sent
    the operator to run an elevated netsh command on a read that never happened.

The module runs at import, so each test execs a fresh copy with subprocess.run
and urlopen patched. Stdlib only: the test-scripts CI job installs pytest,
pytest-timeout and aiohttp and nothing else.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import pathlib
import subprocess
import sys
import urllib.request
from unittest import mock

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "a2a_watchdog.py"

TAILSCALE_IP = "100.64.0.1"
PORT = "8201"


class _Healthy:
    """What urlopen returns for a live health endpoint."""
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Proc:
    """What subprocess.run returns; the watchdog reads .stdout only."""

    def __init__(self, stdout=""):
        self.stdout = stdout


def run_script(tmp_path, ip_result, netsh_result):
    """Exec a fresh copy of the watchdog; return its stdout.

    ``ip_result`` / ``netsh_result`` are either a _Proc or an Exception to raise
    from subprocess.run for that command.
    """
    def fake_run(argv, **kw):
        out = ip_result if argv[0] == "ip" else netsh_result
        if isinstance(out, Exception):
            raise out
        return out

    env = {
        "HOME": str(tmp_path),
        "LOCI_TAILSCALE_IP": TAILSCALE_IP,
        "LOCI_A2A_PORT": PORT,
        "LOCI_PORTPROXY_PS1": str(tmp_path / "portproxy.ps1"),
    }
    out = io.StringIO()
    with mock.patch.dict(os.environ, env, clear=False), \
            mock.patch.object(subprocess, "run", fake_run), \
            mock.patch.object(urllib.request, "urlopen",
                              lambda *a, **k: _Healthy()):
        spec = importlib.util.spec_from_file_location("_a2a_watchdog_uut", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(out):
            spec.loader.exec_module(mod)
    sys.modules.pop("_a2a_watchdog_uut", None)
    return out.getvalue()


def _ip_addr(ip="172.28.1.5"):
    return _Proc(f"    inet {ip}/20 brd 172.28.15.255 scope global eth0\n")


def _netsh(ip="172.28.1.5"):
    return _Proc(f"{TAILSCALE_IP}     {PORT}     {ip}     {PORT}\n")


def test_healthy_and_matching_stays_silent(tmp_path):
    """The contract this all rests on: nothing to say means no output."""
    assert run_script(tmp_path, _ip_addr(), _netsh()) == ""


def test_real_drift_is_still_reported(tmp_path):
    out = run_script(tmp_path, _ip_addr("172.28.9.9"), _netsh("172.28.1.5"))
    assert "WSL IP changed: 172.28.1.5 -> 172.28.9.9" in out
    assert (tmp_path / "portproxy.ps1").exists()


def test_a_genuinely_absent_rule_is_still_reported(tmp_path):
    """netsh answered and listed no rule for our port: that IS 'no rule'."""
    out = run_script(tmp_path, _ip_addr(), _Proc(""))
    assert f"No portproxy rule found for port {PORT}." in out
    assert "run elevated to apply" in out


def test_unreadable_wsl_ip_breaks_the_silence(tmp_path):
    """`ip` missing from PATH under cron skipped both drift branches, and this
    file defines silence as healthy."""
    out = run_script(tmp_path, FileNotFoundError("no ip"), _netsh())
    assert out.strip(), "a probe that never ran must not render as healthy"
    assert "Could not read the WSL eth0 IP" in out
    assert "drift NOT checked" in out


def test_eth0_without_an_inet_address_breaks_the_silence(tmp_path):
    """`ip` ran but eth0 has no IPv4 — also not a reading."""
    out = run_script(tmp_path, _Proc("    link/ether 00:15:5d:00:00:01\n"), _netsh())
    assert "printed no inet address" in out


def test_unreadable_portproxy_does_not_claim_the_rule_is_missing(tmp_path):
    """powershell.exe off PATH must not become the affirmative claim 'there is
    no rule' plus an instruction to go run an elevated netsh command."""
    out = run_script(tmp_path, _ip_addr(), FileNotFoundError("no powershell.exe"))
    assert "Could not read the Windows portproxy rule" in out
    assert f"No portproxy rule found for port {PORT}." not in out
    assert "run elevated to apply" not in out
    assert not (tmp_path / "portproxy.ps1").exists(), \
        "no script should be written on the strength of a read that never happened"
