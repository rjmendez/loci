"""#383 follow-up (f), read side: backfilled tiers reach the firewall readers.

scripts/backfill_provenance_tiers.py appends records to provenance_updates.jsonl
instead of rewriting findings.jsonl. They only matter if the evidence and
candidate readers overlay them, and an explicit tier on the row must still win.
"""
from __future__ import annotations

import json

import pytest

import inv_store
import server


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_store._manifest_cache.clear()
    no = lambda *a, **k: None  # noqa: E731
    for name in ("_qdrant_upsert", "_event_log_append", "_mirror_finding_to_ladybug",
                 "_autolink_finding_to_ladybug"):
        monkeypatch.setattr(server, name, no)
    monkeypatch.setattr(server, "_mnemo_remember", lambda *a, **k: False)
    monkeypatch.setattr(server, "_get_qdrant", lambda *a, **k: (None, None))
    monkeypatch.setattr(server, "_collect_recent_global_audit", lambda *a, **k: [])
    server.investigation_start(investigation_id="bf-case", title="backfill fold")
    yield tmp_path
    inv_store._manifest_cache.clear()


def _store(text, source, **kw):
    r = json.loads(server.investigation_store("bf-case", "inferred", text, source, **kw))
    assert r.get("stored"), r
    return r["finding_id"]


def _backfill(fid, tier, rule="model_writer_source"):
    server._append_jsonl(server._inv_dir("bf-case") / inv_store.PROVENANCE_UPDATES_NAME, {
        "record_type": "provenance_tier", "finding_id": fid, "evidence_provenance_tier": tier,
        "rule": rule, "source": "backfill_provenance_tiers", "ts": "2026-09-24T00:00:00+00:00"})


def _evidence(fid):
    pool, _ = server.build_validation_evidence("bf-case", min_confidence="low")
    return next(e for e in pool if e["text"].startswith("[reasoned]") or fid in e["evidence_id"])


def test_untagged_model_finding_is_defaulted_until_backfilled(store):
    fid = _store("[reasoned] redis pool exhaustion caused the outage", "investigation_reason")
    before = _evidence(fid)
    assert before["provenance_defaulted"] is True
    _backfill(fid, "model_asserted")
    after = _evidence(fid)
    assert after["evidence_provenance_tier"] == "model_asserted"
    assert after["provenance_defaulted"] is False


def test_explicit_tier_beats_a_backfill_record(store):
    fid = _store("[reasoned] explicit", "manual", evidence_provenance_tier="tool_verified")
    _backfill(fid, "model_asserted")
    row = _evidence(fid)
    assert row["evidence_provenance_tier"] == "tool_verified"


def test_load_shows_the_backfilled_tier_and_its_rule(store):
    fid = _store("[reasoned] shown on load", "investigation_reason")
    _backfill(fid, "model_asserted", rule="model_writer_source+reasoned_text_marker")
    out = json.loads(server.investigation_load("bf-case"))
    row = next(f for f in out["recent_findings"] if f["id"] == fid)
    assert row["evidence_provenance_tier"] == "model_asserted"
    assert row["provenance_source"] == "backfill:model_writer_source+reasoned_text_marker"


def test_unknown_tier_in_a_record_is_ignored(store):
    fid = _store("[reasoned] junk record", "investigation_reason")
    _backfill(fid, "definitely_true")
    assert _evidence(fid)["provenance_defaulted"] is True
