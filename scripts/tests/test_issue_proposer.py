import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import issue_proposer
from scripts.issue_proposer import make_loci_dedup_fn, open_proposed_issues, propose_issues


QUEUE = Path(__file__).with_name("_issue_proposals_test_queue.jsonl")


def clean_queue():
    try:
        QUEUE.unlink()
    except FileNotFoundError:
        pass


def finding(title="Reflection loop repeats stale finding", confidence=0.91, evidence=None):
    return {
        "title": title,
        "description": "Reflection loop proposes the same stale remediation on each run.",
        "confidence": confidence,
        "evidence": ["scripts/reflection_loop.py:42 showed the same finding emitted twice"] if evidence is None else evidence,
        "source": "dogfood-reflection-2026-09-17",
    }


def test_dry_run_default_opens_nothing():
    clean_queue()
    result = propose_issues([finding()], "owner/repo", queue_path=QUEUE)
    calls = []

    opened = open_proposed_issues(result["proposals"], lambda **kwargs: calls.append(kwargs))

    assert result["proposals"]
    assert opened["dry_run"] is True
    assert opened["opened"] == []
    assert opened["would_open"][0]["title"] == "Reflection loop repeats stale finding"
    assert calls == []
    clean_queue()


def test_duplicate_proposals_are_suppressed_against_local_queue():
    clean_queue()


def test_duplicate_title_normalization_handles_case_whitespace_and_punctuation():
    clean_queue()
    first = propose_issues([finding(" Reflection   Loop: repeats stale finding!! ")], "owner/repo", queue_path=QUEUE)
    second = propose_issues([finding("reflection-loop repeats stale finding")], "owner/repo", queue_path=QUEUE)

    assert len(first["proposals"]) == 1
    assert second["proposals"] == []
    assert second["deduped"][0]["against"] == "local_queue"
    clean_queue()


def test_near_duplicate_titles_are_suppressed_against_existing_open_issues():
    clean_queue()

    def existing_issues_fn(**_kwargs):
        return [{"title": "Empty JSON parsing crash", "body": "already reported"}]

    result = propose_issues([finding("Crash parsing empty JSON")], "owner/repo", existing_issues_fn=existing_issues_fn, queue_path=QUEUE)

    assert result["proposals"] == []
    assert result["deduped"][0]["against"] == "open_issues"
    assert result["deduped"][0]["method"] == "token_title"
    clean_queue()
    first = propose_issues([finding()], "owner/repo", queue_path=QUEUE)
    second = propose_issues([finding()], "owner/repo", queue_path=QUEUE)

    assert len(first["proposals"]) == 1
    assert second["proposals"] == []
    assert second["deduped"][0]["against"] == "local_queue"
    clean_queue()


def test_duplicate_proposals_are_suppressed_against_existing_open_issues():
    clean_queue()

    def existing_issues_fn(**_kwargs):
        return [{"title": "Reflection loop repeats stale finding", "body": "already open"}]

    result = propose_issues([finding()], "owner/repo", existing_issues_fn=existing_issues_fn, queue_path=QUEUE)

    assert result["proposals"] == []
    assert result["deduped"][0]["against"] == "open_issues"
    clean_queue()


def test_unevidenced_findings_are_dropped():
    clean_queue()
    result = propose_issues([finding(evidence=[])], "owner/repo", queue_path=QUEUE)

    assert result["proposals"] == []
    assert result["dropped"][0]["reason"] == "missing_concrete_evidence"
    clean_queue()


def test_finding_id_without_evidence_is_dropped_not_softened():
    clean_queue()
    raw = finding(evidence=[])
    raw["finding_id"] = "12345678-1234-1234-1234-123456789abc"

    result = propose_issues([raw], "owner/repo", queue_path=QUEUE)

    assert result["proposals"] == []
    assert result["dropped"][0]["reason"] == "missing_concrete_evidence"
    clean_queue()


def test_confidence_floor_is_enforced():
    clean_queue()
    result = propose_issues([finding(confidence=0.5)], "owner/repo", min_confidence=0.8, queue_path=QUEUE)

    assert result["proposals"] == []
    assert result["dropped"][0]["reason"] == "below_confidence_floor"
    clean_queue()


def test_max_proposals_cap_is_enforced_after_ranking():
    clean_queue()
    findings = [
        finding("low accepted second", confidence=0.8),
        finding("top accepted first", confidence=0.95),
        finding("capped candidate", confidence=0.9),
    ]

    result = propose_issues(findings, "owner/repo", min_confidence=0.7, max_proposals=2, queue_path=QUEUE)

    assert [proposal["title"] for proposal in result["proposals"]] == ["top accepted first", "capped candidate"]
    assert result["dropped"][-1]["reason"] == "max_proposals_cap"
    assert result["dropped"][-1]["title"] == "low accepted second"
    clean_queue()


def test_rerun_is_idempotent_with_local_queue():
    clean_queue()
    propose_issues([finding("idempotent queue proposal")], "owner/repo", queue_path=QUEUE)
    second = propose_issues([finding("idempotent queue proposal")], "owner/repo", queue_path=QUEUE)

    assert second["proposals"] == []
    assert second["deduped"]
    assert QUEUE.read_text(encoding="utf-8").count("idempotent queue proposal") == 1
    clean_queue()


def test_gh_fn_raising_exception_fails_open_without_crashing():
    def gh_fn(**_kwargs):
        raise RuntimeError("gh unavailable")

    result = open_proposed_issues([{"repo": "owner/repo", "title": "title", "body": "body"}], gh_fn, confirm=True)

    assert result["fail_open"] is True
    assert result["opened"] == []
    assert "gh unavailable" in result["audit_trail"][0]["error"]


def test_dry_run_cli_with_live_existing_issues_does_not_call_subprocess(monkeypatch, capsys):
    clean_queue()

    def forbidden_run(*_args, **_kwargs):
        raise AssertionError("dry-run must not invoke gh subprocesses")

    monkeypatch.setattr(issue_proposer.subprocess, "run", forbidden_run)
    rc = issue_proposer.main(["--repo", "owner/repo", "--live-existing-issues", "--queue-path", str(QUEUE)])

    assert rc == 0
    captured = capsys.readouterr().out
    assert "live_existing_issues_requires_confirm_open" in captured
    clean_queue()


def test_confirmed_cli_blocks_open_when_proposal_stage_degraded(monkeypatch, capsys):
    clean_queue()
    findings_path = QUEUE.with_name("_issue_proposer_findings.jsonl")
    findings_path.write_text(__import__("json").dumps(finding()) + "\n", encoding="utf-8")
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[:3] == ["gh", "issue", "list"]:
            raise RuntimeError("issue list down")
        raise AssertionError("issue create must be blocked after proposal degradation")

    monkeypatch.setattr(issue_proposer.subprocess, "run", fake_run)
    rc = issue_proposer.main([
        "--repo",
        "owner/repo",
        "--findings-json",
        str(findings_path),
        "--queue-path",
        str(QUEUE),
        "--live-existing-issues",
        "--confirm-open",
    ])

    assert rc == 0
    captured = capsys.readouterr().out
    assert "proposal_fail_open_blocks_confirmed_issue_creation" in captured
    assert any(cmd[:3] == ["gh", "issue", "list"] for cmd in calls)
    assert not any(cmd[:3] == ["gh", "issue", "create"] for cmd in calls)
    findings_path.unlink()
    clean_queue()


def test_dedup_degradation_still_checks_existing_open_issues():
    clean_queue()
    unrelated = {
        "status": "proposed",
        "repo": "owner/repo",
        "title": "unrelated queued proposal",
        "body": "different",
        "confidence": 0.9,
        "evidence": ["scripts/other.py:12 unrelated"],
        "fingerprint": "unrelated",
    }
    QUEUE.write_text(__import__("json").dumps(unrelated) + "\n", encoding="utf-8")

    def degraded_dedup(**_kwargs):
        raise RuntimeError("semantic backend down")

    result = propose_issues(
        [finding("Reflection loop repeats stale finding")],
        "owner/repo",
        existing_issues_fn=lambda **_kwargs: [{"title": "reflection loop repeats stale finding", "body": "already open"}],
        dedup_fn=degraded_dedup,
        queue_path=QUEUE,
    )

    assert result["fail_open"] is True
    assert result["proposals"] == []
    assert result["deduped"][0]["against"] == "open_issues"
    assert any(item.get("action") == "degraded" for item in result["audit_trail"])
    clean_queue()


def test_custom_dedup_fn_can_suppress_semantic_duplicates():
    clean_queue()

    def dedup_fn(**_kwargs):
        return {"duplicate": True, "score": 0.92, "method": "fake_semantic"}

    result = propose_issues(
        [finding("semantic repeat")],
        "owner/repo",
        existing_issues_fn=lambda **_kwargs: [{"title": "Different words", "body": "same meaning"}],
        dedup_fn=dedup_fn,
        queue_path=QUEUE,
    )

    assert result["proposals"] == []
    assert result["deduped"][0]["method"] == "fake_semantic"
    clean_queue()


def test_loci_semantic_dedup_adapter_uses_injected_callable():
    clean_queue()

    def fake_loci_semantic_dedup_fn(**kwargs):
        assert kwargs["threshold"] == 0.88
        return {"clusters": [{"rep_index": 0, "member_indices": [0, 1], "text": "same"}], "degraded": False}

    result = propose_issues(
        [finding("semantic adapter repeat")],
        "owner/repo",
        existing_issues_fn=lambda **_kwargs: [{"title": "Different surface title", "body": "same issue"}],
        dedup_fn=make_loci_dedup_fn(fake_loci_semantic_dedup_fn),
        queue_path=QUEUE,
    )

    assert result["proposals"] == []
    assert result["deduped"][0]["method"].startswith("loci_semantic_dedup")
    clean_queue()


def test_loci_semantic_dedup_degradation_is_audited():
    clean_queue()

    def degraded_loci_semantic_dedup_fn(**_kwargs):
        return {"clusters": [], "degraded": True}

    result = propose_issues(
        [finding("semantic degraded repeat")],
        "owner/repo",
        existing_issues_fn=lambda **_kwargs: [{"title": "Different surface title", "body": "same issue"}],
        dedup_fn=make_loci_dedup_fn(degraded_loci_semantic_dedup_fn),
        queue_path=QUEUE,
    )

    assert result["fail_open"] is True
    assert any("semantic dedup degraded" in item.get("error", "") for item in result["audit_trail"])
    clean_queue()
