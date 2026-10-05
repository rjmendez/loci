"""An investigation records who owns it, and an ownerless one is claimed on its first ACL change.

2026-10-05 audit (S4-01): investigation_start always wrote owner "", and with no owner nothing could be refused an ACL
change, so any local caller could share an investigation with itself, unshare the real users, and take it over. Now the
creator (the transport-bound agent, else this process's HERMES_AGENT_ID) is the owner, and for an investigation created
before owners were recorded the first ACL change claims it.
"""
import json
import sys
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import caller_identity  # noqa: E402
import inv_store  # noqa: E402
import investigation_tools as it  # noqa: E402
import mnemo_ops  # noqa: E402
import server  # noqa: E402


def _j(s):
    return json.loads(s)


@pytest.fixture
def store(tmp_path, monkeypatch):
    mem = tmp_path / "mem"
    mem.mkdir()
    monkeypatch.setattr(server, "MEMORY_DIR", mem)
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(mem))
    monkeypatch.setenv("LOCI_CODE_ROOT", str(tmp_path / "code"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("HERMES_AGENT_ID", raising=False)
    monkeypatch.setattr(inv_store, "_STORE_LOCK_TIMEOUT_S", 0.5)
    inv_store._manifest_cache.clear()
    no = lambda *a, **k: None  # noqa: E731
    for name, value in {"_get_qdrant": lambda *a, **k: (None, None), "_qdrant_upsert": no,
                        "_mnemo_remember": lambda *a, **k: False, "_event_log_append": no,
                        "_mirror_finding_to_ladybug": no, "_autolink_finding_to_ladybug": no}.items():
        monkeypatch.setattr(server, name, value, raising=False)
    monkeypatch.setattr(server, "_get_mnemo_funcs", lambda: (None, None))
    monkeypatch.setattr(mnemo_ops, "_get_mnemo_funcs", lambda: (None, None))
    yield mem
    inv_store._manifest_cache.clear()


def _owner(mem, inv):
    return json.loads((Path(mem) / inv / "manifest.json").read_text())["owner"]


def _acting_as(monkeypatch, who):
    if who:
        monkeypatch.setenv("HERMES_AGENT_ID", who)
    else:
        monkeypatch.delenv("HERMES_AGENT_ID", raising=False)
    inv_store._manifest_cache.clear()


def _denied(out):
    try:
        return json.loads(out).get("error") == "permission_denied"
    except Exception:
        return False


def test_start_records_the_process_identity_as_owner(store, monkeypatch):
    _acting_as(monkeypatch, "rjmendez")
    assert _j(it.investigation_start("case-a", "t")).get("status") == "created"
    assert _owner(store, "case-a") == "rjmendez"


def test_a_transport_bound_identity_takes_precedence(store, monkeypatch):
    _acting_as(monkeypatch, "rjmendez")
    monkeypatch.setattr(caller_identity, "bound_agent_id", lambda: "agent-x")
    it.investigation_start("case-b", "t")
    assert _owner(store, "case-b") == "agent-x"


def test_with_no_identity_at_all_the_owner_stays_empty(store, monkeypatch):
    _acting_as(monkeypatch, None)
    it.investigation_start("case-c", "t")
    assert _owner(store, "case-c") == ""        # documented: give the server HERMES_AGENT_ID to get owners


def test_resuming_never_changes_the_owner(store, monkeypatch):
    _acting_as(monkeypatch, "rjmendez")
    it.investigation_start("case-d", "t")
    _acting_as(monkeypatch, "mallory")
    out = _j(it.investigation_start("case-d", "again"))
    assert out.get("status") == "resumed" or out.get("error") == "permission_denied"
    assert _owner(store, "case-d") == "rjmendez"


def test_a_stranger_cannot_share_unshare_or_take_over_an_owned_investigation(store, monkeypatch):
    _acting_as(monkeypatch, "rjmendez")
    it.investigation_start("case-e", "t")
    assert not _denied(it.investigation_share("case-e", ["bob"]))
    _acting_as(monkeypatch, "mallory")
    assert _denied(it.investigation_share("case-e", ["mallory"]))
    assert _denied(it.investigation_unshare("case-e", ["bob"]))
    manifest = json.loads((Path(store) / "case-e" / "manifest.json").read_text())
    assert manifest["owner"] == "rjmendez" and manifest["acl"] == ["bob"]


def test_the_audit_takeover_no_longer_works_on_an_investigation_without_an_owner(store, monkeypatch):
    """Replay of S4-01: an old investigation (owner ''), the real owner shares it, then an intruder tries to take it."""
    _acting_as(monkeypatch, None)
    it.investigation_start("old-case", "t")
    assert _owner(store, "old-case") == ""
    _acting_as(monkeypatch, "rjmendez")
    out = _j(it.investigation_share("old-case", ["trusted"]))
    assert out.get("owner_claimed") == "rjmendez"
    assert _owner(store, "old-case") == "rjmendez"
    _acting_as(monkeypatch, "intruder")
    assert _denied(it.investigation_unshare("old-case", ["trusted"]))
    assert _denied(it.investigation_share("old-case", ["intruder"]))
    assert _denied(it.investigation_load("old-case"))
    manifest = json.loads((Path(store) / "old-case" / "manifest.json").read_text())
    assert manifest["acl"] == ["trusted"] and manifest["owner"] == "rjmendez"


def test_an_existing_owner_is_never_overwritten_by_a_share(store, monkeypatch):
    _acting_as(monkeypatch, "alice")
    it.investigation_start("case-f", "t")
    out = _j(it.investigation_share("case-f", ["bob"]))
    assert "owner_claimed" not in out and _owner(store, "case-f") == "alice"


def test_an_acl_member_may_still_change_the_acl_without_becoming_owner(store, monkeypatch):
    _acting_as(monkeypatch, "rjmendez")
    it.investigation_start("case-g", "t")
    it.investigation_share("case-g", ["bob"])
    _acting_as(monkeypatch, "bob")
    assert not _denied(it.investigation_share("case-g", ["carol"]))
    assert _owner(store, "case-g") == "rjmendez"


def test_the_requesting_agent_id_argument_cannot_claim_ownership(store, monkeypatch):
    _acting_as(monkeypatch, None)
    it.investigation_start("case-h", "t")
    out = _j(it.investigation_share("case-h", ["x"], requesting_agent_id="mallory"))
    assert "owner_claimed" not in out and _owner(store, "case-h") == ""      # self-declared names never become owner
