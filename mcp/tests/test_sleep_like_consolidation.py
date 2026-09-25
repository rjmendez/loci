"""Tests for memory_consolidate sleep-like phase reporting."""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server  # noqa: E402


class _FakeMnemosyne:
    def sleep_all_sessions(self, dry_run: bool = False):
        return {
            "status": "consolidated",
            "items_consolidated": 2,
            "summaries_created": 1,
            "llm_used": 1,
            "session_results": [{
                "session_id": "sess-1",
                "status": "consolidated",
                "items_consolidated": 2,
                "summaries_created": 1,
                "consolidated_ids": ["wm-1", "wm-2"],
            }],
        }


@pytest.fixture(autouse=True)
def _stable_policy(monkeypatch):
    monkeypatch.setattr(server, "load_state", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        server,
        "consolidation_policy",
        lambda *_args, **_kwargs: {"min_findings_for_causal": 3},
    )
    monkeypatch.setattr(server, "_snapshot_sleep_consolidation_rowid", lambda *_args, **_kwargs: None)


def test_memory_consolidate_reports_sleep_like_phases_when_recent_reasoning_exists(monkeypatch):
    now = datetime.now(timezone.utc).isoformat()
    findings = [
        {"id": "f-1", "record_type": "observed", "source": "investigation_store", "ts": now},
        {"id": "f-2", "record_type": "inferred", "source": "investigation_reason", "ts": now},
        {"id": "f-3", "record_type": "observed", "source": "tool", "ts": now},
    ]

    monkeypatch.setattr(server, "_load_mnemosyne_class", lambda: _FakeMnemosyne)
    monkeypatch.setattr(server, "_find_most_recent_investigation", lambda: ("inv-123", findings))
    monkeypatch.setattr(server, "_run_causal_inference", lambda *_args, **_kwargs: 4)
    monkeypatch.setattr(
        server,
        "_run_consolidation_quality_audit",
        lambda *_args, **_kwargs: {"sampled": 2, "flagged": [], "degraded": False},
    )
    monkeypatch.setattr(server, "_load_reflection_state", lambda: {"last_tick": None})

    parsed = json.loads(server.memory_consolidate(dry_run=False))
    report = parsed["sleep_like_consolidation"]
    phases = report["phases"]

    assert parsed["status"] == "ok"
    assert parsed["causal_edges_inferred"] == 4
    assert parsed["consolidation_quality_audit"] == {"sampled": 2, "flagged": [], "degraded": False}
    assert report["active_burst"] is True
    assert [phase["name"] for phase in phases] == ["summarize", "replay", "stabilize"]
    assert phases[0]["details"]["signals"]["reasoning_findings_recent"] == 1
    provenance = phases[0]["details"]["provenance"]
    assert provenance["aggregation_method"] == "recent_finding_window_with_provenance_fields"
    assert provenance["refs_considered"] == 3
    assert provenance["refs_emitted"] == 3
    assert len(provenance["finding_refs"]) == 3
    assert all("evidence_provenance_tier" in ref for ref in provenance["finding_refs"])
    assert all("provenance_defaulted" in ref for ref in provenance["finding_refs"])
    assert phases[1]["status"] == "ok"
    assert phases[1]["details"]["causal_edges_inferred"] == 4
    assert phases[2]["status"] == "ok"
    assert phases[2]["details"] == {"sampled": 2, "flagged_count": 0, "degraded": False}


def test_memory_consolidate_reports_replay_phase_skipped_without_recent_investigation(monkeypatch):
    monkeypatch.setattr(server, "_load_mnemosyne_class", lambda: _FakeMnemosyne)
    monkeypatch.setattr(server, "_find_most_recent_investigation", lambda: (None, None))
    monkeypatch.setattr(server, "_run_causal_inference", lambda *_args, **_kwargs: 7)
    monkeypatch.setattr(
        server,
        "_run_consolidation_quality_audit",
        lambda *_args, **_kwargs: {"sampled": 0, "flagged": [], "degraded": True},
    )
    monkeypatch.setattr(server, "_load_reflection_state", lambda: {"last_tick": None})

    parsed = json.loads(server.memory_consolidate(dry_run=False))
    report = parsed["sleep_like_consolidation"]
    replay_phase = next(phase for phase in report["phases"] if phase["name"] == "replay")

    assert parsed["causal_edges_inferred"] == 0
    assert report["active_burst"] is False
    assert replay_phase["status"] == "skipped"
    assert replay_phase["details"]["findings_considered"] == 0
    assert replay_phase["details"]["min_findings_required"] == 3


def test_memory_consolidate_fail_closed_when_sleep_like_report_violates_contract(monkeypatch):
    monkeypatch.setattr(server, "_load_mnemosyne_class", lambda: _FakeMnemosyne)
    monkeypatch.setattr(server, "_find_most_recent_investigation", lambda: ("inv-123", []))
    monkeypatch.setattr(server, "_run_causal_inference", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        server,
        "_run_consolidation_quality_audit",
        lambda *_args, **_kwargs: {"sampled": 0, "flagged": [], "degraded": False},
    )
    # Break the report's input (a dependency of the builder), not the checker:
    # a finding ref without provenance fields must be refused by the REAL gate.
    real_summary = server._sleep_like_burst_summary

    def _summary_missing_provenance(*args, **kwargs):
        out = real_summary(*args, **kwargs)
        out["provenance"]["finding_refs"] = [{"finding_id": "f-1"}]
        return out

    monkeypatch.setattr(server, "_sleep_like_burst_summary", _summary_missing_provenance)

    parsed = json.loads(server.memory_consolidate(dry_run=False))

    assert parsed["status"] == "error"
    assert parsed["type"] == "ValueError"


def _valid_report():
    """A report built by the real builder; the positive control for every case below."""
    now = datetime.now(timezone.utc).isoformat()
    findings = [{"id": "f-1", "record_type": "observed", "source": "tool", "ts": now}]
    return server._sleep_like_consolidation_report(
        investigation_id="inv-1", findings=findings, min_findings_for_causal=1,
        causal_edges_inferred=0, quality_audit={"sampled": 1, "flagged": [], "degraded": False},
    )


def test_sleep_like_invariants_accept_a_well_formed_report(monkeypatch):
    monkeypatch.setattr(server, "_load_reflection_state", lambda: {"last_tick": None})
    report = _valid_report()
    assert [p["name"] for p in report["phases"]] == ["summarize", "replay", "stabilize"]
    assert server._assert_sleep_like_consolidation_invariants(report) is None


def _set(path, value):
    def _mutate(report):
        target = report
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return _mutate


@pytest.mark.parametrize("mutate, match", [
    (lambda r: r.pop("active_burst"), "active_burst must be bool"),
    (_set(["active_burst"], "yes"), "active_burst must be bool"),
    (_set(["phases"], []), "phases must contain summarize/replay/stabilize"),
    (lambda r: r["phases"].reverse(), "phase order mismatch"),
    (_set(["phases", 0, "status"], "skipped"), "invalid status for summarize: 'skipped'"),
    (_set(["phases", 1, "status"], "failed"), "invalid status for replay: 'failed'"),
    (_set(["phases", 2, "details"], None), "stabilize details must be dict"),
    (lambda r: r["phases"][0]["details"].pop("provenance"), "summarize.provenance missing"),
    (_set(["phases", 0, "details", "provenance", "finding_refs"], "f-1"),
     "finding_refs must be list"),
    (_set(["phases", 0, "details", "provenance", "finding_refs"], ["f-1"]),
     "finding ref must be object"),
    (_set(["phases", 0, "details", "provenance", "finding_refs"], [{"evidence_provenance_tier": "x"}]),
     "finding ref missing provenance fields"),
])
def test_sleep_like_invariants_reject_each_contract_violation(monkeypatch, mutate, match):
    monkeypatch.setattr(server, "_load_reflection_state", lambda: {"last_tick": None})
    report = _valid_report()
    mutate(report)
    with pytest.raises(ValueError, match=re.escape(match)):
        server._assert_sleep_like_consolidation_invariants(report)


def test_sleep_like_invariants_reject_a_non_dict_report():
    with pytest.raises(ValueError, match="report must be a dict"):
        server._assert_sleep_like_consolidation_invariants(["not", "a", "dict"])


def test_sleep_like_summary_reduces_reflection_noise_to_snapshot(monkeypatch):
    now = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(
        server,
        "_load_reflection_state",
        lambda: {
            "last_tick": {
                "ts": now,
                "processed_items": 4,
                "findings_written": 2,
                "investigation_id": "inv-123",
                "queue": [{"path": "very/noisy.log"}],
                "processed": {"very/noisy.log": "2026-09-22T00:00:00Z"},
            },
        },
    )

    out = server._sleep_like_burst_summary("inv-123", [])
    tick = out["reflection_last_tick"]

    assert tick == {
        "ts": now,
        "processed_items": 4,
        "findings_written": 2,
        "investigation_id": "inv-123",
    }


def test_memory_consolidate_does_not_overwrite_reflection_state(monkeypatch):
    monkeypatch.setattr(server, "_load_mnemosyne_class", lambda: _FakeMnemosyne)
    monkeypatch.setattr(server, "_find_most_recent_investigation", lambda: (None, None))
    monkeypatch.setattr(server, "_run_causal_inference", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        server,
        "_run_consolidation_quality_audit",
        lambda *_args, **_kwargs: {"sampled": 0, "flagged": [], "degraded": False},
    )
    monkeypatch.setattr(
        server,
        "_load_reflection_state",
        lambda: {"last_tick": {"ts": datetime.now(timezone.utc).isoformat(), "processed_items": 1, "findings_written": 1}},
    )
    monkeypatch.setattr(
        server,
        "_save_reflection_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not write reflection state")),
    )

    parsed = json.loads(server.memory_consolidate(dry_run=False))
    assert parsed["status"] == "ok"
