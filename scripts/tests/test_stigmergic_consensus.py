from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import stigmergic_consensus as C  # noqa: E402

NOW = 1_700_000_000_000


def _finding(subtask, answer, *, agent="a", confidence="high", refs=None, created_ms=NOW, **extra):
    row = {
        "subtask": subtask,
        "answer": answer,
        "confidence": confidence,
        "ok": True,
        "parse_ok": True,
        "agent_id": agent,
        "created_ms": created_ms,
    }
    if refs is not None:
        row["refs"] = refs
    row.update(extra)
    return row


def test_accepts_corroborated_resourceful_claim_without_model_calls():
    findings = [
        _finding("Check mcp/server.py auth", "Auth is enforced because mcp/server.py:42 verifies tokens.", agent="seed-1"),
        _finding("Verify mcp/server.py auth", "Auth is enforced because mcp/server.py:42 verifies tokens.", agent="seed-2"),
    ]

    out = C.consensus_gate(findings, C.StigmergicConfig(enabled=True), now_ms=NOW)

    assert out["model_calls"] == 0
    assert out["flagged_indices"] == []
    assert out["accepted_indices"] == [0, 1]
    assert out["clusters"][0]["state"] == "accepted"
    assert out["clusters"][0]["trail"] >= 0.60
    assert out["clusters"][0]["resource"] >= 0.50


def test_degraded_single_plausible_claim_escalates():
    findings = [_finding("Check cache TTL", "Cache TTL appears to be five minutes.", agent="seed-1")]

    out = C.apply_stigmergic_consensus(findings, {"flagged_indices": [], "escalation_reasons": {}}, C.StigmergicConfig(enabled=True), now_ms=NOW)

    assert out["triage"]["flagged_indices"] == [0]
    assert out["consensus"]["clusters"][0]["state"] == "degraded"
    assert "stigmergic:under_corroborated" in out["triage"]["escalation_reasons"]["0"]


def test_alarm_refutation_escalates_whole_cluster():
    findings = [
        _finding("Check auth policy", "Auth requires MFA and is supported by docs/auth.md:12.", agent="seed-1"),
        _finding("Check auth policy", "This refutes the claim: auth does not require MFA in docs/auth.md:12.", agent="seed-2", votes=[{"voter_id": "seed-2", "vote": "refute", "confidence": 0.9}]),
    ]

    out = C.consensus_gate(findings, C.StigmergicConfig(enabled=True), now_ms=NOW)

    assert out["flagged_indices"] == [0, 1]
    assert out["clusters"][0]["state"] == "alarm"
    assert out["clusters"][0]["refute_ratio"] > 1 / 3


def test_decay_and_revision_boundary_prevent_stale_acceptance():
    old = NOW - 90 * 60 * 1000
    findings = [
        _finding("Check source", "Evidence says yes in src/app.py:10.", agent="seed-1", created_ms=old, revision_id="old"),
        _finding("Check source", "Evidence says yes in src/app.py:10.", agent="seed-2", created_ms=old, revision_id="old"),
    ]
    cfg = C.StigmergicConfig(enabled=True, ttl_minutes=60, revision_id="new")

    out = C.consensus_gate(findings, cfg, now_ms=NOW)

    assert out["flagged_indices"] == [0, 1]
    assert out["clusters"][0]["state"] == "no_quorum"
    assert "ttl_expired" in out["clusters"][0]["reasons"]
    assert "revision_boundary" in out["clusters"][0]["reasons"]


def test_same_evidence_groups_different_claim_words():
    findings = [
        _finding("Is the API guarded?", "Yes, see src/api.py:88 token check.", agent="seed-1"),
        _finding("Do handlers verify callers?", "Yes, see src/api.py:88 token check.", agent="seed-2"),
    ]

    out = C.consensus_gate(findings, C.StigmergicConfig(enabled=True), now_ms=NOW)

    assert out["stats"]["cluster_count"] == 1
    assert out["clusters"][0]["state"] == "accepted"
