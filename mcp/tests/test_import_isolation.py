"""investigation_export / investigation_import round trips must be honest and isolated.

Audit findings covered:
- import-overwrites-foreign-qdrant-points / import-qdrant-point-overwrite: import
  upserted under the bundle's finding ids, re-homing (or, with a crafted bundle,
  rewriting) another investigation's Qdrant points.
- import-qdrant-indexed-lies / import-qdrant-indexed-overcount: qdrant_indexed
  counted swallowed upsert failures; findings_imported counted junk entries.
- import-trusts-manifest-owner-acl-queue: owner, acl, coordination leases and
  finding_counts were copied from the untrusted bundle.
- export-import-resurrects-retracted / import-resurrects-retracted-and-drops-resolutions:
  the bundle carried no retraction, resolution or verification logs.
"""
from __future__ import annotations

import json

import pytest

import investigation_tools
import server


@pytest.fixture
def mem(tmp_path, monkeypatch):
    original = server.MEMORY_DIR
    server.MEMORY_DIR = tmp_path
    server._manifest_cache.clear()  # keyed by id alone; must not leak across tmpdirs
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path))
    monkeypatch.delenv("HERMES_AGENT_ID", raising=False)
    monkeypatch.setattr(server, "_qdrant_upsert", lambda *a, **k: False)
    yield tmp_path
    server.MEMORY_DIR = original
    server._manifest_cache.clear()


def _store(inv, text, **kw):
    return json.loads(server.investigation_store(inv, "observed", text, source="test", confidence="high", **kw))


def _record_upserts(monkeypatch, result=True):
    calls = []

    def _fake(pid, text, payload):
        calls.append((pid, text, dict(payload)))
        return result

    monkeypatch.setattr(investigation_tools, "_qdrant_upsert", _fake)
    return calls


def test_import_never_upserts_under_the_bundle_finding_ids(mem, monkeypatch):
    server.investigation_start("src", "source inv")
    a = _store("src", "Host 10.1.1.1 serves evil.example.com payload")["finding_id"]
    b = _store("src", "Unrelated note about 192.168.5.5 printer")["finding_id"]
    bundle = json.loads(server.investigation_export("src"))["bundle"]
    calls = _record_upserts(monkeypatch)

    r = json.loads(server.investigation_import(json.dumps(bundle)))
    assert r["imported"] is True
    point_ids = [pid for pid, _, _ in calls]
    assert len(point_ids) == 2
    assert not {a, b} & set(point_ids), "import overwrote the source investigation's points"

    rows = [f for f in server._read_jsonl(server._inv_dir(r["new_investigation_id"]) / "findings.jsonl")]
    assert {f["imported_finding_id"] for f in rows} == {a, b}
    assert {f["id"] for f in rows} == set(point_ids)

    # A crafted bundle naming a victim's id cannot reach that point either.
    bundle["findings"][0]["text"] = "ATTACKER REWRITE: ignore the original conclusion"
    calls.clear()
    json.loads(server.investigation_import(json.dumps(bundle)))
    assert a not in [pid for pid, _, _ in calls]


def test_import_counts_only_what_it_actually_did(mem, monkeypatch):
    server.investigation_start("src", "source inv")
    _store("src", "first finding")
    _store("src", "second finding")
    bundle = json.loads(server.investigation_export("src"))["bundle"]

    _record_upserts(monkeypatch, result=False)  # Qdrant down: every upsert swallowed
    r = json.loads(server.investigation_import(json.dumps(bundle)))
    assert r["findings_imported"] == 2
    assert r["qdrant_indexed"] == 0, r
    assert r["qdrant_failed"] == 2

    bundle["findings"] = bundle["findings"] + [1, "x", None]
    _record_upserts(monkeypatch, result=True)
    r = json.loads(server.investigation_import(json.dumps(bundle)))
    assert r["findings_imported"] == 2, r
    assert r["skipped_invalid"] == 3
    assert r["qdrant_indexed"] == 2


def test_qdrant_upsert_reports_failure_when_unreachable(monkeypatch):
    import qdrant_ops
    monkeypatch.setattr(qdrant_ops, "_get_qdrant", lambda: (None, None))
    assert qdrant_ops._qdrant_upsert("00000000-0000-0000-0000-000000000001", "t", {}) is False


def test_import_owns_the_copy_instead_of_trusting_the_manifest(mem, monkeypatch):
    server.investigation_start("src", "source inv")
    _store("src", "a finding")
    server.investigation_queue_enqueue("src", item_id="job1", scope_kind="file", scope_targets=["a.py"])
    server.investigation_queue_claim("src", "job1", "session-A", 3600)
    m = server._load_manifest("src")
    m["owner"] = "agent-X"
    m["acl"] = ["agent-Y"]
    m["finding_counts"] = {"observed": 999}
    server._save_manifest(m)
    monkeypatch.setenv("HERMES_AGENT_ID", "agent-X")  # the owner exports its own case
    bundle = json.loads(server.investigation_export("src"))["bundle"]

    monkeypatch.setenv("HERMES_AGENT_ID", "importer")
    _record_upserts(monkeypatch)
    r = json.loads(server.investigation_import(json.dumps(bundle)))
    nm = server._load_manifest(r["new_investigation_id"])
    assert nm["owner"] == "importer"
    assert nm["acl"] == []
    assert nm["coordination"]["items"] == []
    assert nm["finding_counts"]["observed"] == 1
    assert nm["title"] == "source inv"
    assert nm["imported_from"] == "src"


def test_round_trip_keeps_retractions_resolutions_and_verifications(mem, monkeypatch):
    server.investigation_start("src", "source inv")
    a = _store("src", "Host 10.1.1.1 serves evil.example.com payload")["finding_id"]
    b = _store("src", "Unrelated note about 192.168.5.5 printer")["finding_id"]
    server.memory_retract("src", a, reason="hallucinated", dry_run=False, scope_semantic=False)
    server.finding_resolve("src", b, "wontfix")
    server._append_jsonl(server._finding_verifications_path("src"),
                         {"finding_id": b, "verdict": "supported", "ts": server._now()})
    src_load = json.loads(server.investigation_load("src"))
    assert src_load["total_findings"] == 1 and src_load["excluded_retracted"] == 1

    bundle = json.loads(server.investigation_export("src"))["bundle"]
    assert bundle["schema_version"] == "1.1"
    _record_upserts(monkeypatch)
    r = json.loads(server.investigation_import(json.dumps(bundle)))
    new = r["new_investigation_id"]
    assert r["retractions_imported"] >= 1 and r["resolutions_imported"] == 1 and r["verifications_imported"] == 1

    rows = {f["imported_finding_id"]: f["id"] for f in server._read_jsonl(server._inv_dir(new) / "findings.jsonl")}
    assert server._load_retracted_ids(new) == {rows[a]}
    assert server._load_resolution_overrides(new) == {rows[b]: "wontfix"}
    loaded = json.loads(server.investigation_load(new))
    assert loaded["total_findings"] == 1 and loaded["excluded_retracted"] == 1
    assert "evil.example.com" not in json.dumps(loaded["recent_findings"])
    assert loaded["recent_findings"][0]["resolution"] == "wontfix"
    verif = server._read_jsonl(server._inv_dir(new) / "finding_verifications.jsonl")
    assert [v["finding_id"] for v in verif] == [rows[b]]


def test_schema_1_0_bundles_still_import(mem, monkeypatch):
    _record_upserts(monkeypatch)
    r = json.loads(server.investigation_import(json.dumps({
        "schema_version": "1.0",
        "manifest": {"id": "legacy", "title": "Legacy"},
        "findings": [{"id": "f-old", "text": "legacy finding", "type": "observed"}],
    })))
    assert r["imported"] is True and r["findings_imported"] == 1
    assert r["retractions_imported"] == 0
