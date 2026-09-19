from __future__ import annotations

import importlib.util
import json
import pathlib
import sys


SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "endpoint_maintenance.py"


def _load():
    spec = importlib.util.spec_from_file_location("endpoint_maintenance_uut", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    sys.modules.pop(spec.name, None)
    return mod


def _run_main(mod, capsys, argv: list[str], results):
    queue = list(results)

    def fake_run_ssh(endpoint: str, script: str, timeout: int):
        assert endpoint == "endpoint-test"
        assert isinstance(script, str) and script
        assert timeout == 9
        assert queue, "script invoked _run_ssh more times than expected"
        return queue.pop(0)

    mod._run_ssh = fake_run_ssh
    rc = mod.main(["--endpoint", "endpoint-test", "--timeout", "9", *argv])
    payload = json.loads(capsys.readouterr().out)
    assert not queue, "not all queued fake SSH results were consumed"
    return rc, payload


def test_healthy_sync_health_and_services(capsys):
    mod = _load()
    rc, payload = _run_main(
        mod,
        capsys,
        [
            "--health-url",
            "http://127.0.0.1:8201/health",
            "--service",
            "a2a.service",
            "--service",
            "context-bridge.service",
        ],
        [
            mod.CommandResult(
                returncode=0,
                stdout="CLONED=0\nSHA=abc123\nCLEAN=1\nDIRTY_COUNT=0\n",
                stderr="",
            ),
            mod.CommandResult(
                returncode=0,
                stdout="PRESENT=1\nTOOL=curl\nHTTP_STATUS=200\n",
                stderr="",
            ),
            mod.CommandResult(
                returncode=0,
                stdout=(
                    "PRESENT=1\nLOAD_STATE=loaded\nACTIVE_STATE=active\n"
                    "SUB_STATE=running\nUNIT_FILE_STATE=enabled\n"
                ),
                stderr="",
            ),
            mod.CommandResult(
                returncode=0,
                stdout=(
                    "PRESENT=1\nLOAD_STATE=loaded\nACTIVE_STATE=active\n"
                    "SUB_STATE=running\nUNIT_FILE_STATE=enabled\n"
                ),
                stderr="",
            ),
        ],
    )

    assert rc == 0
    assert payload["ok"] is True
    assert payload["errors"] == []
    assert payload["sync"]["cloned"] is False
    assert payload["sync"]["remote_sha"] == "abc123"
    assert payload["sync"]["clean"] is True
    assert payload["sync"]["dirty_count"] == 0
    assert payload["health"]["present"] is True
    assert payload["health"]["http_status"] == 200
    assert payload["health"]["ok"] is True
    assert payload["health"]["tool"] == "curl"
    assert [svc["name"] for svc in payload["services"]] == [
        "a2a.service",
        "context-bridge.service",
    ]
    assert all(svc["ok"] is True for svc in payload["services"])


def test_sync_failure_is_critical_and_captures_error(capsys):
    mod = _load()
    rc, payload = _run_main(
        mod,
        capsys,
        ["--health-url", ""],
        [
            mod.CommandResult(
                returncode=17,
                stdout="",
                stderr="git clone failed: permission denied",
            ),
        ],
    )

    assert rc == 1
    assert payload["ok"] is False
    assert payload["sync"]["remote_sha"] is None
    assert payload["sync"]["cloned"] is None
    assert payload["sync"]["clean"] is None
    assert payload["sync"]["dirty_count"] is None
    assert payload["errors"] == [
        {
            "critical": True,
            "message": "git clone failed: permission denied",
            "step": "sync",
        }
    ]


def test_endpoint_unreachable_reports_ssh_failures(capsys):
    mod = _load()
    err = "ssh: connect to host endpoint-test port 22: No route to host"
    rc, payload = _run_main(
        mod,
        capsys,
        ["--service", "a2a.service"],
        [
            mod.CommandResult(returncode=255, stdout="", stderr=err),
            mod.CommandResult(returncode=255, stdout="", stderr=err),
            mod.CommandResult(returncode=255, stdout="", stderr=err),
        ],
    )

    assert rc == 1
    assert payload["ok"] is False
    assert payload["services"] == [{"name": "a2a.service", "present": None, "error": err}]
    assert payload["health"]["checked"] is True
    assert payload["health"]["present"] is None
    assert payload["errors"] == [
        {"step": "sync", "message": err, "critical": True},
        {"step": "health", "message": err, "critical": False},
        {"step": "services", "message": f"a2a.service: {err}", "critical": False},
    ]


def test_json_output_shape_invariants(capsys):
    mod = _load()
    rc, payload = _run_main(
        mod,
        capsys,
        ["--health-url", "", "--service", "context-bridge.service"],
        [
            mod.CommandResult(returncode=0, stdout="CLONED=1\nSHA=def456\nCLEAN=0\nDIRTY_COUNT=2\n", stderr=""),
            mod.CommandResult(returncode=0, stdout="PRESENT=0\nREASON=unit_not_found\n", stderr=""),
        ],
    )

    assert rc == 1
    assert set(payload.keys()) == {"endpoint", "ok", "sync", "health", "services", "errors"}
    assert payload["endpoint"] == "endpoint-test"
    assert isinstance(payload["ok"], bool)
    assert set(payload["sync"].keys()) == {
        "repo_path",
        "repo_url",
        "branch",
        "attempted",
        "cloned",
        "remote_sha",
        "clean",
        "dirty_count",
    }
    assert set(payload["health"].keys()) == {"checked", "url", "present", "http_status", "ok"}
    assert isinstance(payload["services"], list)
    assert isinstance(payload["errors"], list)
