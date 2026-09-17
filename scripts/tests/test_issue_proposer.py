import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import issue_proposer
from scripts.issue_proposer import (
    make_loci_dedup_fn,
    open_proposed_issues,
    propose_issues,
    promote_proposal,
    record_proposal_rejection,
)


QUEUE = Path(__file__).with_name("_issue_proposals_test_queue.jsonl")
DASHBOARD = Path(__file__).with_name("_issue_proposals_test_dashboard.md")
REJECTIONS = Path(__file__).with_name("_issue_proposals_test_rejections.jsonl")
FINDINGS = Path(__file__).with_name("_issue_proposer_findings.jsonl")


def clean_files():
    for path in (QUEUE, DASHBOARD, REJECTIONS, FINDINGS):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def finding(title="Reflection loop repeats stale finding", confidence=0.91, evidence=None, **extra):
    base = {
        "title": title,
        "description": "Reflection loop proposes the same stale remediation on each run.",
        "confidence": confidence,
        "evidence": ["scripts/reflection_loop.py:42 stale finding emitted twice"] if evidence is None else evidence,
        "source": "dogfood-reflection-2026-09-17",
    }
    base.update(extra)
    return base


def test_dashboard_is_default_output_and_contains_proposal_id():
    clean_files()
    result = propose_issues([finding()], "owner/repo", queue_path=QUEUE, dashboard_path=DASHBOARD, rejection_path=REJECTIONS)

    assert result["dry_run"] is True
    assert result["proposals"]
    proposal_id = result["proposals"][0]["proposal_id"]
    dashboard = DASHBOARD.read_text(encoding="utf-8")
    assert "Loci issue proposal dashboard" in dashboard
    assert proposal_id in dashboard
    assert "Reflection loop repeats stale finding" in dashboard
    clean_files()


def test_promotion_requires_both_confirm_and_specific_proposal_id():
    clean_files()
    result = propose_issues([finding()], "owner/repo", queue_path=QUEUE, dashboard_path=DASHBOARD, rejection_path=REJECTIONS)
    proposal = result["proposals"][0]
    calls = []

    no_id = open_proposed_issues([proposal], lambda **kwargs: calls.append(kwargs), confirm=True, allowed_repos={"owner/repo"})
    no_confirm = open_proposed_issues([proposal], lambda **kwargs: calls.append(kwargs), confirm=False, promotion_id=proposal["proposal_id"], allowed_repos={"owner/repo"})
    confirmed = open_proposed_issues([proposal], lambda **kwargs: calls.append(kwargs) or {"url": "https://example.invalid/1"}, confirm=True, promotion_id=proposal["proposal_id"], allowed_repos={"owner/repo"})

    assert no_id["audit_trail"][0]["reason"] == "promotion_id_required"
    assert no_confirm["dry_run"] is True
    assert confirmed["opened"] == [{"url": "https://example.invalid/1"}]
    assert len(calls) == 1
    clean_files()


def test_allowlist_blocks_confirmed_promotion_without_network():
    clean_files()
    result = propose_issues([finding()], "owner/repo", queue_path=QUEUE, dashboard_path=DASHBOARD, rejection_path=REJECTIONS)
    proposal_id = result["proposals"][0]["proposal_id"]
    calls = []

    promoted = promote_proposal(
        proposal_id,
        queue_path=QUEUE,
        dashboard_path=DASHBOARD,
        rejection_path=REJECTIONS,
        gh_fn=lambda **kwargs: calls.append(kwargs),
        confirm=True,
        allowed_repos=[],
    )

    assert promoted["dry_run"] is True
    assert promoted["audit_trail"][-1]["reason"] == "repo_not_allowlisted"
    assert calls == []
    clean_files()


def test_fingerprint_is_evidence_keyed_not_prose_keyed():
    clean_files()
    first = finding(
        title="First wording",
        description="Body wording one.",
        evidence=[{"file": "pkg/worker.py", "line": 88, "symbol": "run", "signature": "ValueError: stale task"}],
    )
    second = finding(
        title="Completely different wording",
        description="Body wording two.",
        evidence=[{"symbol": "run", "signature": "ValueError: stale task", "line": 88, "file": "pkg/worker.py"}],
    )

    first_result = propose_issues([first], "owner/repo", queue_path=QUEUE, dashboard_path=DASHBOARD, rejection_path=REJECTIONS)
    second_result = propose_issues([second], "owner/repo", queue_path=QUEUE, dashboard_path=DASHBOARD, rejection_path=REJECTIONS)

    assert first_result["proposals"][0]["fingerprint"] == first_result["proposals"][0]["proposal_id"]
    assert second_result["proposals"] == []
    assert second_result["deduped"][0]["method"] == "fingerprint"
    clean_files()


def test_rejection_ledger_persists_and_blocks_reproposal():
    clean_files()
    first = propose_issues([finding()], "owner/repo", queue_path=QUEUE, dashboard_path=DASHBOARD, rejection_path=REJECTIONS)
    proposal_id = first["proposals"][0]["proposal_id"]

    rejected = record_proposal_rejection(
        proposal_id,
        queue_path=QUEUE,
        dashboard_path=DASHBOARD,
        rejection_path=REJECTIONS,
        reason="not actionable",
    )
    second = propose_issues([finding(title="new prose same evidence")], "owner/repo", queue_path=QUEUE, dashboard_path=DASHBOARD, rejection_path=REJECTIONS)

    assert rejected["recorded"][0]["proposal_id"] == proposal_id
    assert "not actionable" in REJECTIONS.read_text(encoding="utf-8")
    assert second["proposals"] == []
    assert second["dropped"][0]["reason"] == "previously_rejected"
    assert "Rejected proposals" in DASHBOARD.read_text(encoding="utf-8")
    clean_files()


def test_near_duplicate_titles_are_suppressed_against_existing_open_issues():
    clean_files()

    def existing_issues_fn(**_kwargs):
        return [{"title": "Empty JSON parsing crash", "body": "already reported"}]

    result = propose_issues([finding("Crash parsing empty JSON")], "owner/repo", existing_issues_fn=existing_issues_fn, queue_path=QUEUE)

    assert result["proposals"] == []
    assert result["deduped"][0]["against"] == "open_issues"
    assert result["deduped"][0]["method"] == "token_title"
    clean_files()
    first = propose_issues([finding()], "owner/repo", queue_path=QUEUE)
    second = propose_issues([finding()], "owner/repo", queue_path=QUEUE)

    assert len(first["proposals"]) == 1
    assert second["proposals"] == []
    assert second["deduped"][0]["against"] == "local_queue"
    clean_files()


def test_duplicate_proposals_are_suppressed_against_existing_open_issues():
    clean_files()

    def existing_issues_fn(**_kwargs):
        return [{"title": "Reflection loop repeats stale finding", "body": "already open"}]

    result = propose_issues([finding()], "owner/repo", existing_issues_fn=existing_issues_fn, queue_path=QUEUE)

    assert result["proposals"] == []
    assert result["deduped"][0]["against"] == "open_issues"
    clean_files()


def test_unevidenced_findings_are_dropped():
    clean_files()
    result = propose_issues([finding(evidence=[])], "owner/repo", queue_path=QUEUE)

    assert result["proposals"] == []
    assert result["dropped"][0]["reason"] == "missing_concrete_evidence"
    clean_files()


def test_prose_only_evidence_is_rejected():
    clean_files()
    result = propose_issues(
        [finding(evidence=["model says this is bad and very confident about it"])],
        "owner/repo",
        queue_path=QUEUE,
    )

    assert result["proposals"] == []
    assert result["dropped"][0]["reason"] == "missing_concrete_evidence"
    clean_files()


def test_finding_id_style_evidence_is_accepted():
    clean_files()
    result = propose_issues(
        [finding(evidence=["duplicate remediation flagged as FND-4821"])],
        "owner/repo",
        queue_path=QUEUE,
    )

    assert len(result["proposals"]) == 1
    clean_files()


def test_structured_evidence_object_with_id_field_is_accepted():
    clean_files()
    result = propose_issues(
        [finding(evidence=[{"id": "9f86d081884c7d659a2feaa0c55ad015"}])],
        "owner/repo",
        queue_path=QUEUE,
    )

    assert len(result["proposals"]) == 1
    clean_files()


def test_finding_id_without_evidence_is_dropped_not_softened():
    raw = finding(evidence=[])
    raw["finding_id"] = "12345678-1234-1234-1234-123456789abc"

    result = propose_issues([raw], "owner/repo", queue_path=QUEUE)

    assert result["proposals"] == []
    assert result["dropped"][0]["reason"] == "missing_concrete_evidence"
    clean_files()


def test_fingerprint_is_stable_across_reworded_title_and_confidence():
    clean_files()
    same_evidence = ["scripts/reflection_loop.py:42 showed the same finding emitted twice"]
    first = propose_issues([finding("Original phrasing of the finding", confidence=0.9, evidence=same_evidence)], "owner/repo", queue_path=QUEUE)
    second = propose_issues(
        [finding("A totally reworded description of the same bug", confidence=0.75, evidence=same_evidence)],
        "owner/repo",
        queue_path=QUEUE,
    )

    assert len(first["proposals"]) == 1
    assert second["proposals"] == []
    assert second["deduped"][0]["method"] == "fingerprint"
    clean_files()


def test_confidence_floor_is_enforced():
    clean_files()
    result = propose_issues([finding(confidence=0.5)], "owner/repo", min_confidence=0.8, queue_path=QUEUE)

    assert result["proposals"] == []
    assert result["dropped"][0]["reason"] == "below_confidence_floor"
    clean_files()


def test_max_proposals_cap_is_enforced_after_ranking():
    clean_files()
    findings = [
        finding("low accepted second", confidence=0.8, evidence=["scripts/low.py:10 low finding"]),
        finding("top accepted first", confidence=0.95, evidence=["scripts/top.py:20 top finding"]),
        finding("capped candidate", confidence=0.9, evidence=["scripts/capped.py:30 capped finding"]),
    ]

    result = propose_issues(findings, "owner/repo", min_confidence=0.7, max_proposals=2, queue_path=QUEUE)

    assert [proposal["title"] for proposal in result["proposals"]] == ["top accepted first", "capped candidate"]
    assert result["dropped"][-1]["reason"] == "max_proposals_cap"
    assert result["dropped"][-1]["title"] == "low accepted second"
    clean_files()


def test_rerun_is_idempotent_with_local_queue():
    clean_files()
    propose_issues([finding("idempotent queue proposal")], "owner/repo", queue_path=QUEUE)
    second = propose_issues([finding("idempotent queue proposal")], "owner/repo", queue_path=QUEUE)

    assert second["proposals"] == []
    assert second["deduped"]
    assert QUEUE.read_text(encoding="utf-8").count("idempotent queue proposal") == 1
    clean_files()


def test_gh_fn_raising_exception_fails_open_without_crashing():
    def gh_fn(**_kwargs):
        raise RuntimeError("gh unavailable")

    result = open_proposed_issues(
        [{"repo": "owner/repo", "title": "title", "body": "body", "proposal_id": "abc123"}],
        gh_fn,
        confirm=True,
        promotion_id="abc123",
        allowed_repos={"owner/repo"},
    )

    assert result["fail_open"] is True
    assert result["opened"] == []
    assert "gh unavailable" in result["audit_trail"][0]["error"]


def test_dry_run_cli_with_live_existing_issues_does_not_call_subprocess(monkeypatch, capsys):
    clean_files()

    def forbidden_run(*_args, **_kwargs):
        raise AssertionError("dry-run must not invoke gh subprocesses")

    monkeypatch.setattr(issue_proposer.subprocess, "run", forbidden_run)
    rc = issue_proposer.main(["--repo", "owner/repo", "--live-existing-issues", "--queue-path", str(QUEUE)])

    assert rc == 0
    captured = capsys.readouterr().out
    assert "live_existing_issues_requires_confirm_open" in captured
    clean_files()


def test_dry_run_cli_with_live_existing_issues_and_promotion_never_calls_subprocess(monkeypatch, capsys):
    clean_files()
    seed = propose_issues([finding()], "owner/repo", queue_path=QUEUE, dashboard_path=DASHBOARD, rejection_path=REJECTIONS)
    proposal_id = seed["proposals"][0]["proposal_id"]

    def forbidden_run(*_args, **_kwargs):
        raise AssertionError("confirm=false paths must not invoke subprocess")

    monkeypatch.setattr(issue_proposer.subprocess, "run", forbidden_run)
    rc = issue_proposer.main([
        "--repo",
        "owner/repo",
        "--queue-path",
        str(QUEUE),
        "--dashboard-path",
        str(DASHBOARD),
        "--rejection-path",
        str(REJECTIONS),
        "--promote-proposal",
        proposal_id,
        "--live-existing-issues",
    ])

    assert rc == 0
    captured = capsys.readouterr().out
    assert "live_existing_issues_requires_confirm_open" in captured
    clean_files()


def test_confirmed_promotion_checks_existing_open_issues_before_create(monkeypatch, capsys):
    clean_files()
    FINDINGS.write_text(json.dumps(finding()) + "\n", encoding="utf-8")
    seeded = propose_issues([finding()], "owner/repo", queue_path=QUEUE, dashboard_path=DASHBOARD, rejection_path=REJECTIONS)
    proposal_id = seeded["proposals"][0]["proposal_id"]
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[:3] == ["gh", "issue", "list"]:
            raise RuntimeError("issue list down")
        raise AssertionError("issue create must not run after lookup degradation")

    monkeypatch.setattr(issue_proposer.subprocess, "run", fake_run)
    rc = issue_proposer.main([
        "--repo",
        "owner/repo",
        "--queue-path",
        str(QUEUE),
        "--dashboard-path",
        str(DASHBOARD),
        "--rejection-path",
        str(REJECTIONS),
        "--promote-proposal",
        proposal_id,
        "--confirm-open",
        "--allow-repo",
        "owner/repo",
        "--live-existing-issues",
    ])

    assert rc == 0
    captured = capsys.readouterr().out
    assert "promotion_fail_open_blocks_issue_creation" in captured
    assert any(cmd[:3] == ["gh", "issue", "list"] for cmd in calls)
    assert not any(cmd[:3] == ["gh", "issue", "create"] for cmd in calls)
    clean_files()


def test_confidence_floor_and_cap_are_preserved():
    clean_files()
    findings = [
        finding("low accepted second", confidence=0.8),
        finding("top accepted first", confidence=0.95, evidence=["scripts/a.py:1 repeated failure"]),
        finding("capped candidate", confidence=0.9, evidence=["scripts/b.py:2 repeated failure"]),
        finding("too low", confidence=0.4, evidence=["scripts/c.py:3 repeated failure"]),
    ]

    result = propose_issues(findings, "owner/repo", min_confidence=0.7, max_proposals=2, queue_path=QUEUE, dashboard_path=DASHBOARD, rejection_path=REJECTIONS)

    assert [proposal["title"] for proposal in result["proposals"]] == ["top accepted first", "capped candidate"]
    assert any(item["reason"] == "below_confidence_floor" for item in result["dropped"])
    assert any(item["reason"] == "max_proposals_cap" for item in result["dropped"])
    clean_files()


def test_dedup_degradation_still_checks_existing_open_issues():
    clean_files()
    QUEUE.write_text(json.dumps({
        "status": "proposed",
        "proposal_id": "unrelated",
        "repo": "owner/repo",
        "title": "unrelated queued proposal",
        "body": "different",
        "confidence": 0.9,
        "evidence": ["scripts/other.py:12 unrelated"],
        "fingerprint": "unrelated",
    }) + "\n", encoding="utf-8")

    def degraded_dedup(**_kwargs):
        raise RuntimeError("semantic backend down")

    result = propose_issues(
        [finding("Reflection loop repeats stale finding")],
        "owner/repo",
        existing_issues_fn=lambda **_kwargs: [{"title": "reflection loop repeats stale finding", "body": "already open"}],
        dedup_fn=degraded_dedup,
        queue_path=QUEUE,
        dashboard_path=DASHBOARD,
        rejection_path=REJECTIONS,
    )

    assert result["fail_open"] is True
    assert result["proposals"] == []
    assert result["deduped"][0]["against"] == "open_issues"
    assert any(item.get("action") == "degraded" for item in result["audit_trail"])
    clean_files()


def test_loci_semantic_dedup_adapter_uses_evidence_text():
    clean_files()

    def fake_loci_semantic_dedup_fn(**kwargs):
        assert kwargs["threshold"] == 0.88
        assert "scripts/reflection_loop.py:42 stale finding emitted twice" in kwargs["items"][0]["text"]
        return {"clusters": [{"rep_index": 0, "member_indices": [0, 1], "text": "same"}], "degraded": False}

    result = propose_issues(
        [finding("semantic adapter repeat")],
        "owner/repo",
        existing_issues_fn=lambda **_kwargs: [{"title": "Different surface title", "body": "same issue", "evidence": ["scripts/reflection_loop.py:42 stale finding emitted twice"]}],
        dedup_fn=make_loci_dedup_fn(fake_loci_semantic_dedup_fn),
        queue_path=QUEUE,
        dashboard_path=DASHBOARD,
        rejection_path=REJECTIONS,
    )

    assert result["proposals"] == []
    assert result["deduped"][0]["method"].startswith("loci_semantic_dedup")
    clean_files()
