"""The suite is isolated from the machine it runs on.

conftest.py calls testsupport/loci_hermetic.install() before any test module --
and so before ``server`` -- is imported. These tests prove the effect end to end:
a real investigation_store call lands in the session temp root, the endpoints
are unreachable, and an access to the real ~/.loci or ~/.hermes is refused and
reported rather than silently performed.
"""
import json
import os
import sqlite3
import uuid
from pathlib import Path

import pytest

import loci_hermetic

pytestmark = pytest.mark.skipif(
    not loci_hermetic.installed(),
    reason=f"{loci_hermetic.OPT_OUT_VAR}=1: isolation deliberately disabled",
)


def _root() -> Path:
    return Path(loci_hermetic.root()).resolve()


def _inside_root(path) -> bool:
    return Path(path).resolve().is_relative_to(_root())


def test_home_config_and_memory_dir_are_in_the_temp_root():
    import backends
    import server

    assert _inside_root(Path.home())
    assert _inside_root(os.environ["LOCI_MEMORY_DIR"])
    assert _inside_root(os.environ["MNEMOSYNE_DATA_DIR"])
    assert _inside_root(backends._CONFIG_PATH)
    assert not Path(backends._CONFIG_PATH).exists()
    assert _inside_root(server.MEMORY_DIR)


def test_backends_resolve_to_an_unreachable_endpoint():
    import server  # noqa: F401  (import runs load_env(), which must not refill these)

    for key in ("QDRANT_URL", "OLLAMA_BASE_URL", "VLLM_BASE_URL"):
        assert os.environ[key] == loci_hermetic.UNREACHABLE, key
    assert "QDRANT_API_KEY" not in os.environ


def test_investigation_store_writes_into_the_temp_root():
    import server
    from investigation_tools import investigation_start

    inv = f"hermetic-{uuid.uuid4().hex[:8]}"
    investigation_start(inv, "hermetic isolation probe")
    # "observed", not "observation": the store rejects an unknown finding_type, and
    # this probe then passed on the directory investigation_start had already made.
    stored = json.loads(server.investigation_store(
        inv, "observed", "isolation probe finding", "test_hermetic_isolation",
    ))
    assert "error" not in stored, stored

    hits = [p for p in _root().rglob("*") if inv in str(p)]
    assert hits, f"nothing for {inv} under {_root()}"
    assert all(_inside_root(p) for p in hits)
    findings = [p for p in hits if p.name == "findings.jsonl"]
    assert len(findings) == 1
    assert [json.loads(line)["id"] for line in findings[0].read_text().splitlines()] == [
        stored["finding_id"]]


@pytest.mark.parametrize("opener", ["open", "sqlite", "mkdir"])
def test_access_to_a_real_store_is_refused_and_recorded(opener):
    live = Path(loci_hermetic.forbidden_roots()[0]) / f"hermetic-probe-{uuid.uuid4().hex}"
    start = loci_hermetic.violation_count()
    with pytest.raises(loci_hermetic.LiveStoreAccessError):
        if opener == "open":
            open(live / "state.json", "w")  # noqa: SIM115
        elif opener == "sqlite":
            sqlite3.connect(str(live / "store.db"))
        else:
            os.mkdir(live)
    recorded = loci_hermetic.take_violations(start)
    assert len(recorded) == 1 and str(live) in recorded[0]
    assert not live.exists()


def test_the_repo_env_files_are_not_loaded(tmp_path):
    import dotenv

    repo = Path(loci_hermetic.REPO_ROOT)
    assert dotenv.load_dotenv(repo / ".env") is False
    assert dotenv.load_dotenv(repo / "mcp" / ".env", override=True) is False

    # An explicit, test-owned file still loads.
    env = tmp_path / ".env"
    env.write_text("LOCI_HERMETIC_PROBE=1\n")
    try:
        assert dotenv.load_dotenv(env) is True
        assert os.environ["LOCI_HERMETIC_PROBE"] == "1"
    finally:
        os.environ.pop("LOCI_HERMETIC_PROBE", None)


def test_the_event_log_is_written_inside_the_temp_root():
    """Pin the leak that inflated the live event log.

    Before this harness, 26 test files pointed LOCI_MEMORY_DIR at a temp dir and
    none redirected the event log. Test findings went to the temp dir and were
    deleted with it. Their "store" events went to the live
    ~/.hermes/event_log.jsonl and stayed. By 2026-09-24 the live log held about
    43,000 store events for about 1,800 investigations that never existed on
    disk. That is why it showed 46,798 stores against about 5,790 findings. Both
    event-log writers must resolve inside the temp root, and a real store must
    land its event there.
    """
    import sys

    import server
    from investigation_tools import investigation_start

    scripts_dir = str(Path(loci_hermetic.REPO_ROOT) / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import event_log
    import route_audit

    log = Path(event_log._DEFAULT_LOG)
    assert _inside_root(log)
    assert _inside_root(route_audit._default_log_path())

    start = loci_hermetic.violation_count()
    inv = f"hermetic-evlog-{uuid.uuid4().hex[:8]}"
    investigation_start(inv, "event log isolation probe")
    stored = json.loads(server.investigation_store(
        inv, "observed", "event log isolation probe finding", "test_hermetic_isolation",
    ))
    assert "error" not in stored, stored
    assert loci_hermetic.take_violations(start) == []

    events = [json.loads(line) for line in log.read_text().splitlines()]
    mine = [e for e in events if e.get("investigation_id") == inv]
    assert [(e["op"], e["finding_id"]) for e in mine] == [("store", stored["finding_id"])]


def test_gpu_load_signal_is_redirected_into_the_temp_root():
    import gpu_load

    path = gpu_load._signal_path()
    assert os.environ[loci_hermetic.GPU_LOAD_PATH_VAR] == str(path)
    assert _inside_root(path)
    assert not path.exists()
    assert path != gpu_load.live_signal_path()


def test_generate_sees_no_gpu_load_under_test():
    """generate() reads the signal on every call; under test it must find none.

    Before the redirect, a busy workstation GPU made _read_gpu_load() return a
    loaded snapshot and changed routing and timeouts in test_llm_local*.py.
    """
    import llm_local

    start = loci_hermetic.violation_count()
    assert llm_local._read_gpu_load() is None
    assert loci_hermetic.take_violations(start) == []


@pytest.mark.skipif(not loci_hermetic.forbidden_files(), reason="no live GPU signal path on this OS")
def test_reading_the_live_gpu_load_signal_is_refused_and_recorded(monkeypatch):
    import gpu_load

    live = gpu_load.live_signal_path()
    assert str(live) in loci_hermetic.forbidden_files()
    # Undo the redirect, as a regression would: the read must be refused and recorded.
    monkeypatch.delenv(loci_hermetic.GPU_LOAD_PATH_VAR)
    start = loci_hermetic.violation_count()
    assert gpu_load.read_gpu_load() is None  # fail-open: the refusal is swallowed
    recorded = loci_hermetic.take_violations(start)
    assert len(recorded) == 1 and str(live) in recorded[0]
