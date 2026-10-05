"""Focused tests for memory_consolidate's advisory consolidation quality audit."""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server  # noqa: E402


def _make_fake_mnemosyne(*, summary_of="wm-1,wm-2", merged_text="summary",
                         result=None):
    class FakeMnemosyne:
        def __init__(self):
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            conn.execute(
                "CREATE TABLE working_memory (id TEXT PRIMARY KEY, content TEXT, source TEXT, timestamp TEXT)"
            )
            conn.execute(
                "CREATE TABLE episodic_memory (id TEXT, content TEXT, source TEXT, session_id TEXT, summary_of TEXT)"
            )
            conn.executemany(
                "INSERT INTO working_memory (id, content, source, timestamp) VALUES (?, ?, ?, ?)",
                [
                    ("wm-1", "Alice rotated the AWS key after the incident.", "notes", "2026-09-16T00:00:00Z"),
                    ("wm-2", "Bob disabled the compromised CI runner pending rebuild.", "notes", "2026-09-16T00:05:00Z"),
                ],
            )
            conn.commit()
            self.beam = type("Beam", (), {"conn": conn})()
            self.sleep_calls: list[bool] = []

        def sleep_all_sessions(self, dry_run: bool = False):
            self.sleep_calls.append(dry_run)
            if not dry_run:
                self.beam.conn.execute(
                    "INSERT INTO episodic_memory (id, content, source, session_id, summary_of) VALUES (?, ?, ?, ?, ?)",
                    ("ep-1", merged_text, "sleep_consolidation", "sess-1", summary_of),
                )
                self.beam.conn.commit()
            return result or {
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

    return FakeMnemosyne


@pytest.fixture(autouse=True)
def _reset_audit_gen_fn():
    original = server._consolidation_quality_audit_gen_fn
    yield
    server._consolidation_quality_audit_gen_fn = original


def test_memory_consolidate_fail_open_when_merge_details_cannot_be_determined(monkeypatch):
    monkeypatch.setattr(
        server,
        "_load_mnemosyne_class",
        lambda: _make_fake_mnemosyne(summary_of=""),
    )
    monkeypatch.setattr(server, "_find_most_recent_investigation", lambda: (None, []))

    parsed = json.loads(server.memory_consolidate(dry_run=False))

    assert parsed["status"] == "ok"
    assert parsed["dry_run"] is False
    assert parsed["causal_edges_inferred"] == 0
    assert parsed["consolidation_quality_audit"] == {
        "sampled": 0,
        "flagged": [],
        "degraded": True,
    }


def test_memory_consolidate_fail_open_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(server, "_load_mnemosyne_class", lambda: _make_fake_mnemosyne())
    monkeypatch.setattr(server, "_find_most_recent_investigation", lambda: (None, []))
    prompts: list[str] = []

    def _unavailable_model(prompt, *, fmt=None, max_tokens=256):
        prompts.append(prompt)
        return {"text": "", "ok": False, "why": "no Ollama endpoint resolved"}

    server._consolidation_quality_audit_gen_fn = _unavailable_model

    parsed = json.loads(server.memory_consolidate(dry_run=False))

    assert parsed["status"] == "ok"
    assert parsed["consolidation_quality_audit"] == {
        "sampled": 0,
        "flagged": [],
        "degraded": True,
    }
    # The sampler found the merge and the configured model WAS asked about it:
    # "degraded" here means the model failed, not that nothing was sampled.
    assert len(prompts) == 1
    assert "Alice rotated the AWS key after the incident." in prompts[0]
    assert "Bob disabled the compromised CI runner pending rebuild." in prompts[0]


def _scripted_model(verdict: str, concern: str, calls: list):
    def _gen(prompt, *, fmt=None, max_tokens=256):
        calls.append({"prompt": prompt, "fmt": fmt})
        return {
            "ok": True,
            "text": json.dumps({"verdict": verdict, "concern": concern, "confidence": 0.8}),
        }
    return _gen


def test_memory_consolidate_quality_audit_positive_control_preserved(monkeypatch):
    """Positive twin of the fail-open tests (same fixture): a working model
    samples the one real merge and reports it clean, not degraded."""
    monkeypatch.setattr(server, "_load_mnemosyne_class", lambda: _make_fake_mnemosyne(
        merged_text="Alice rotated the AWS key; Bob disabled the CI runner."))
    monkeypatch.setattr(server, "_find_most_recent_investigation", lambda: (None, []))
    calls: list = []
    server._consolidation_quality_audit_gen_fn = _scripted_model("preserved", "", calls)

    parsed = json.loads(server.memory_consolidate(dry_run=False))

    assert parsed["consolidation_quality_audit"] == {
        "sampled": 1,
        "flagged": [],
        "degraded": False,
    }
    assert len(calls) == 1
    assert calls[0]["fmt"] == "json"
    assert "MERGED RESULT:\nAlice rotated the AWS key; Bob disabled the CI runner." in calls[0]["prompt"]
    stabilize = parsed["sleep_like_consolidation"]["phases"][2]
    assert stabilize == {"name": "stabilize", "status": "ok",
                         "details": {"sampled": 1, "flagged_count": 0, "degraded": False}}


def test_memory_consolidate_quality_audit_positive_control_flags_lost_merge(monkeypatch):
    monkeypatch.setattr(server, "_load_mnemosyne_class", lambda: _make_fake_mnemosyne(
        merged_text="Alice rotated the AWS key."))
    monkeypatch.setattr(server, "_find_most_recent_investigation", lambda: (None, []))
    calls: list = []
    server._consolidation_quality_audit_gen_fn = _scripted_model(
        "lost_or_conflated", "Drops the CI runner being disabled.", calls)

    parsed = json.loads(server.memory_consolidate(dry_run=False))

    assert parsed["consolidation_quality_audit"] == {
        "sampled": 1,
        "flagged": [{
            "summary": "Alice rotated the AWS key.",
            "concern": "Drops the CI runner being disabled.",
            "session_id": "sess-1",
            "source_entry_ids": ["wm-1", "wm-2"],
            "source_entry_count": 2,
        }],
        "degraded": False,
    }
    assert len(calls) == 1


def test_memory_consolidate_dry_run_shape_and_behavior_are_unchanged(monkeypatch):
    fake_cls = _make_fake_mnemosyne()
    monkeypatch.setattr(server, "_load_mnemosyne_class", lambda: fake_cls)
    monkeypatch.setattr(server, "_find_most_recent_investigation", lambda: (None, []))
    monkeypatch.setattr(
        server,
        "_run_consolidation_quality_audit",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("audit must not run on dry_run")),
    )

    parsed = json.loads(server.memory_consolidate(dry_run=True))

    # consolidation_aggregation was added to EVERY memory_consolidate response
    # (dry-run included) by the slow-neuromodulation work in 46fcf49 and is
    # pinned by test_slow_neuromodulation. It is deterministic provenance for the
    # causal threshold, not audit output, so it is checked separately here; what
    # this test guards is that the quality audit adds nothing to a dry run and
    # the pre-existing fields are untouched.
    aggregation = parsed.pop("consolidation_aggregation")
    assert aggregation["method"] == "deterministic_sleep_summary_plus_slow_threshold"
    assert aggregation["default_min_findings_for_causal"] == 3
    assert isinstance(aggregation["applied_min_findings_for_causal"], int)
    assert aggregation["modulation"]["provenance"] == "deterministic_derived"
    assert "consolidation_quality_audit" not in parsed

    assert parsed == {
        "status": "ok",
        "dry_run": True,
        "result": {
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
        },
        "causal_edges_inferred": 0,
    }


def test_consolidation_quality_flags_include_provenance_and_are_deduped(monkeypatch):
    sample = {
        "session_id": "sess-1",
        "merged_summary": "The incident was handled.",
        "source_entries": [
            {"id": "wm-1", "content": "Alice rotated the AWS key."},
            {"id": "wm-2", "content": "Bob disabled the compromised CI runner."},
        ],
    }
    monkeypatch.setattr(server, "_fetch_consolidation_quality_samples", lambda *_a, **_k: ([sample, dict(sample)], False))
    monkeypatch.setattr(
        "consolidation_quality_audit.audit_merge_quality",
        lambda *_a, **_k: {
            "available": True,
            "verdict": "lost_or_conflated",
            "concern": "The merged result omits distinct remediation actions.",
            "confidence": 0.91,
            "degraded": False,
            "error": "",
        },
    )

    parsed = server._run_consolidation_quality_audit(
        m=object(),
        result={"items_consolidated": 2, "session_results": [{"session_id": "sess-1"}]},
        baseline_rowid=None,
    )

    assert parsed is not None
    assert parsed["sampled"] == 2
    assert len(parsed["flagged"]) == 1
    flagged = parsed["flagged"][0]
    assert flagged["session_id"] == "sess-1"
    assert flagged["source_entry_ids"] == ["wm-1", "wm-2"]
    assert flagged["source_entry_count"] == 2
