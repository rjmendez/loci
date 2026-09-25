import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import issue_proposer
from scripts.issue_proposer import (
    make_loci_dedup_fn,
    open_proposed_issues,
    propose_issues,
    promote_proposal,
    record_proposal_rejection,
)


_PRODUCTION_FILES = (
    issue_proposer.DEFAULT_QUEUE_PATH,
    issue_proposer.DEFAULT_DASHBOARD_PATH,
    issue_proposer.DEFAULT_REJECTION_PATH,
)


def _stat(path):
    try:
        st = Path(path).stat()
    except FileNotFoundError:
        return None
    return (st.st_size, st.st_mtime_ns)


@pytest.fixture(autouse=True)
def _production_files_untouched():
    """The queue, dashboard and rejection ledger default to files next to the
    script, i.e. the operator's real ones. Every test passes tmp paths; this
    fails any test that falls back to a default (a write) instead."""
    before = {path: _stat(path) for path in _PRODUCTION_FILES}
    yield
    after = {path: _stat(path) for path in _PRODUCTION_FILES}
    assert after == before, "a test wrote the production issue-proposer files"


@pytest.fixture(autouse=True)
def gh_calls(monkeypatch):
    """Record every subprocess call instead of running gh.

    A stub that raised AssertionError here was swallowed by the fail-open
    `except Exception` in _call_existing_issues, so "gh was called" passed.
    Tests assert on this list instead.
    """
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        stdout = "[]" if cmd[:3] == ["gh", "issue", "list"] else "https://example.invalid/new\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(issue_proposer.subprocess, "run", fake_run)
    return calls


@pytest.fixture
def files(tmp_path):
    return SimpleNamespace(
        queue=tmp_path / "queue.jsonl",
        dashboard=tmp_path / "dashboard.md",
        rejections=tmp_path / "rejections.jsonl",
        findings=tmp_path / "findings.jsonl",
    )


def paths(files):
    return {"queue_path": files.queue, "dashboard_path": files.dashboard, "rejection_path": files.rejections}


def cli_paths(files):
    return ["--queue-path", str(files.queue), "--dashboard-path", str(files.dashboard),
            "--rejection-path", str(files.rejections)]


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


def test_dashboard_is_default_output_and_contains_proposal_id(files):
    result = propose_issues([finding()], "owner/repo", **paths(files))

    assert result["dry_run"] is True
    assert result["proposals"]
    proposal_id = result["proposals"][0]["proposal_id"]
    dashboard = files.dashboard.read_text(encoding="utf-8")
    assert "Loci issue proposal dashboard" in dashboard
    assert proposal_id in dashboard
    assert "Reflection loop repeats stale finding" in dashboard


def test_promotion_requires_both_confirm_and_specific_proposal_id(files):
    result = propose_issues([finding()], "owner/repo", **paths(files))
    proposal = result["proposals"][0]
    calls = []

    no_id = open_proposed_issues([proposal], lambda **kwargs: calls.append(kwargs), confirm=True, allowed_repos={"owner/repo"})
    no_confirm = open_proposed_issues([proposal], lambda **kwargs: calls.append(kwargs), confirm=False, promotion_id=proposal["proposal_id"], allowed_repos={"owner/repo"})
    confirmed = open_proposed_issues([proposal], lambda **kwargs: calls.append(kwargs) or {"url": "https://example.invalid/1"}, confirm=True, promotion_id=proposal["proposal_id"], allowed_repos={"owner/repo"})

    assert no_id["audit_trail"][0]["reason"] == "promotion_id_required"
    assert no_confirm["dry_run"] is True
    assert confirmed["opened"] == [{"url": "https://example.invalid/1"}]
    assert len(calls) == 1


def test_allowlist_blocks_confirmed_promotion_without_network(files):
    result = propose_issues([finding()], "owner/repo", **paths(files))
    proposal_id = result["proposals"][0]["proposal_id"]
    calls = []

    promoted = promote_proposal(
        proposal_id,
        **paths(files),
        gh_fn=lambda **kwargs: calls.append(kwargs),
        confirm=True,
        allowed_repos=[],
    )

    assert promoted["dry_run"] is True
    assert promoted["audit_trail"][-1]["reason"] == "repo_not_allowlisted"
    assert calls == []


def test_fingerprint_is_evidence_keyed_not_prose_keyed(files):
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

    first_result = propose_issues([first], "owner/repo", **paths(files))
    second_result = propose_issues([second], "owner/repo", **paths(files))

    assert first_result["proposals"][0]["fingerprint"] == first_result["proposals"][0]["proposal_id"]
    assert second_result["proposals"] == []
    assert second_result["deduped"][0]["method"] == "fingerprint"


def test_rejection_ledger_persists_and_blocks_reproposal(files):
    first = propose_issues([finding()], "owner/repo", **paths(files))
    proposal_id = first["proposals"][0]["proposal_id"]

    rejected = record_proposal_rejection(proposal_id, **paths(files), reason="not actionable")
    second = propose_issues([finding(title="new prose same evidence")], "owner/repo", **paths(files))

    assert rejected["recorded"][0]["proposal_id"] == proposal_id
    assert "not actionable" in files.rejections.read_text(encoding="utf-8")
    assert second["proposals"] == []
    assert second["dropped"][0]["reason"] == "previously_rejected"
    assert "Rejected proposals" in files.dashboard.read_text(encoding="utf-8")


def test_near_duplicate_titles_are_suppressed_against_existing_open_issues(files, tmp_path):
    def existing_issues_fn(**_kwargs):
        return [{"title": "Empty JSON parsing crash", "body": "already reported"}]

    result = propose_issues([finding("Crash parsing empty JSON")], "owner/repo", existing_issues_fn=existing_issues_fn, **paths(files))

    assert result["proposals"] == []
    assert result["deduped"][0]["against"] == "open_issues"
    assert result["deduped"][0]["method"] == "token_title"

    fresh = SimpleNamespace(queue=tmp_path / "q2.jsonl", dashboard=tmp_path / "d2.md",
                            rejections=tmp_path / "r2.jsonl")
    first = propose_issues([finding()], "owner/repo", **paths(fresh))
    second = propose_issues([finding()], "owner/repo", **paths(fresh))

    assert len(first["proposals"]) == 1
    assert second["proposals"] == []
    assert second["deduped"][0]["against"] == "local_queue"


def test_duplicate_proposals_are_suppressed_against_existing_open_issues(files):
    def existing_issues_fn(**_kwargs):
        return [{"title": "Reflection loop repeats stale finding", "body": "already open"}]

    result = propose_issues([finding()], "owner/repo", existing_issues_fn=existing_issues_fn, **paths(files))

    assert result["proposals"] == []
    assert result["deduped"][0]["against"] == "open_issues"


def test_unevidenced_findings_are_dropped(files):
    result = propose_issues([finding(evidence=[])], "owner/repo", **paths(files))

    assert result["proposals"] == []
    assert result["dropped"][0]["reason"] == "missing_concrete_evidence"


def test_prose_only_evidence_is_rejected(files):
    result = propose_issues(
        [finding(evidence=["model says this is bad and very confident about it"])],
        "owner/repo",
        **paths(files),
    )

    assert result["proposals"] == []
    assert result["dropped"][0]["reason"] == "missing_concrete_evidence"


def test_finding_id_style_evidence_is_accepted(files):
    result = propose_issues(
        [finding(evidence=["duplicate remediation flagged as FND-4821"])],
        "owner/repo",
        **paths(files),
    )

    assert len(result["proposals"]) == 1


def test_structured_evidence_object_with_id_field_is_accepted(files):
    result = propose_issues(
        [finding(evidence=[{"id": "9f86d081884c7d659a2feaa0c55ad015"}])],
        "owner/repo",
        **paths(files),
    )

    assert len(result["proposals"]) == 1


def test_finding_id_without_evidence_is_dropped_not_softened(files):
    raw = finding(evidence=[])
    raw["finding_id"] = "12345678-1234-1234-1234-123456789abc"

    result = propose_issues([raw], "owner/repo", **paths(files))

    assert result["proposals"] == []
    assert result["dropped"][0]["reason"] == "missing_concrete_evidence"


def test_fingerprint_is_stable_across_reworded_title_and_confidence(files):
    same_evidence = ["scripts/reflection_loop.py:42 showed the same finding emitted twice"]
    first = propose_issues([finding("Original phrasing of the finding", confidence=0.9, evidence=same_evidence)], "owner/repo", **paths(files))
    second = propose_issues(
        [finding("A totally reworded description of the same bug", confidence=0.75, evidence=same_evidence)],
        "owner/repo",
        **paths(files),
    )

    assert len(first["proposals"]) == 1
    assert second["proposals"] == []
    assert second["deduped"][0]["method"] == "fingerprint"


def test_confidence_floor_is_enforced(files):
    result = propose_issues([finding(confidence=0.5)], "owner/repo", min_confidence=0.8, **paths(files))

    assert result["proposals"] == []
    assert result["dropped"][0]["reason"] == "below_confidence_floor"


def test_max_proposals_cap_is_enforced_after_ranking(files):
    findings = [
        finding("low accepted second", confidence=0.8, evidence=["scripts/low.py:10 low finding"]),
        finding("top accepted first", confidence=0.95, evidence=["scripts/top.py:20 top finding"]),
        finding("capped candidate", confidence=0.9, evidence=["scripts/capped.py:30 capped finding"]),
    ]

    result = propose_issues(findings, "owner/repo", min_confidence=0.7, max_proposals=2, **paths(files))

    assert [proposal["title"] for proposal in result["proposals"]] == ["top accepted first", "capped candidate"]
    assert result["dropped"][-1]["reason"] == "max_proposals_cap"
    assert result["dropped"][-1]["title"] == "low accepted second"


def test_rerun_is_idempotent_with_local_queue(files):
    propose_issues([finding("idempotent queue proposal")], "owner/repo", **paths(files))
    second = propose_issues([finding("idempotent queue proposal")], "owner/repo", **paths(files))

    assert second["proposals"] == []
    assert second["deduped"]
    assert files.queue.read_text(encoding="utf-8").count("idempotent queue proposal") == 1


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


def test_dry_run_cli_with_live_existing_issues_does_not_call_subprocess(files, gh_calls, capsys):
    files.findings.write_text(json.dumps(finding()) + "\n", encoding="utf-8")
    rc = issue_proposer.main(["--repo", "owner/repo", "--live-existing-issues",
                              "--findings-json", str(files.findings), *cli_paths(files)])

    assert rc == 0
    assert gh_calls == []
    out = json.loads(capsys.readouterr().out)
    assert out["proposal_result"]["audit_trail"][-1] == {
        "action": "skipped", "reason": "live_existing_issues_requires_confirm_open"}
    assert len(out["proposal_result"]["proposals"]) == 1
    assert out["open_result"]["dry_run"] is True
    assert out["open_result"]["opened"] == []


def test_dry_run_cli_with_live_existing_issues_and_promotion_never_calls_subprocess(files, gh_calls, capsys):
    seed = propose_issues([finding()], "owner/repo", **paths(files))
    proposal_id = seed["proposals"][0]["proposal_id"]

    rc = issue_proposer.main([
        "--repo", "owner/repo", *cli_paths(files),
        "--promote-proposal", proposal_id,
        "--live-existing-issues",
    ])

    assert rc == 0
    assert gh_calls == []
    out = json.loads(capsys.readouterr().out)
    promotion = out["promotion_result"]
    assert promotion["dry_run"] is True
    assert promotion["audit_trail"][-1] == {"action": "skipped", "reason": "live_existing_issues_requires_confirm_open"}


def test_confirmed_promotion_with_live_lookup_creates_the_issue(files, gh_calls, capsys):
    # Positive twin of the dry-run tests: with --confirm-open and an allowlisted
    # repo, the live lookup runs first and then the issue is created.
    seed = propose_issues([finding()], "owner/repo", **paths(files))
    proposal_id = seed["proposals"][0]["proposal_id"]

    rc = issue_proposer.main([
        "--repo", "owner/repo", *cli_paths(files),
        "--promote-proposal", proposal_id,
        "--confirm-open", "--allow-repo", "owner/repo",
        "--live-existing-issues",
    ])

    assert rc == 0
    assert [cmd[:3] for cmd in gh_calls] == [["gh", "issue", "list"], ["gh", "issue", "create"]]
    assert gh_calls[1][gh_calls[1].index("--repo") + 1] == "owner/repo"
    assert gh_calls[1][gh_calls[1].index("--title") + 1] == "Reflection loop repeats stale finding"
    promotion = json.loads(capsys.readouterr().out)["promotion_result"]
    assert promotion["opened"] == [{"url": "https://example.invalid/new"}]


def test_confirmed_promotion_checks_existing_open_issues_before_create(files, monkeypatch, capsys):
    files.findings.write_text(json.dumps(finding()) + "\n", encoding="utf-8")
    seeded = propose_issues([finding()], "owner/repo", **paths(files))
    proposal_id = seeded["proposals"][0]["proposal_id"]
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[:3] == ["gh", "issue", "list"]:
            raise RuntimeError("issue list down")
        return subprocess.CompletedProcess(cmd, 0, stdout="https://example.invalid/should-not-exist\n", stderr="")

    monkeypatch.setattr(issue_proposer.subprocess, "run", fake_run)
    rc = issue_proposer.main([
        "--repo", "owner/repo", *cli_paths(files),
        "--promote-proposal", proposal_id,
        "--confirm-open", "--allow-repo", "owner/repo",
        "--live-existing-issues",
    ])

    assert rc == 0
    captured = capsys.readouterr().out
    assert "promotion_fail_open_blocks_issue_creation" in captured
    assert [cmd[:3] for cmd in calls] == [["gh", "issue", "list"]]


def test_confidence_floor_and_cap_are_preserved(files):
    findings = [
        finding("low accepted second", confidence=0.8),
        finding("top accepted first", confidence=0.95, evidence=["scripts/a.py:1 repeated failure"]),
        finding("capped candidate", confidence=0.9, evidence=["scripts/b.py:2 repeated failure"]),
        finding("too low", confidence=0.4, evidence=["scripts/c.py:3 repeated failure"]),
    ]

    result = propose_issues(findings, "owner/repo", min_confidence=0.7, max_proposals=2, **paths(files))

    assert [proposal["title"] for proposal in result["proposals"]] == ["top accepted first", "capped candidate"]
    assert any(item["reason"] == "below_confidence_floor" for item in result["dropped"])
    assert any(item["reason"] == "max_proposals_cap" for item in result["dropped"])


def test_dedup_degradation_still_checks_existing_open_issues(files):
    files.queue.write_text(json.dumps({
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
        **paths(files),
    )

    assert result["fail_open"] is True
    assert result["proposals"] == []
    assert result["deduped"][0]["against"] == "open_issues"
    assert any(item.get("action") == "degraded" for item in result["audit_trail"])


def test_loci_semantic_dedup_adapter_uses_evidence_text(files):
    seen = []

    def fake_loci_semantic_dedup_fn(**kwargs):
        seen.append(kwargs)
        return {"clusters": [{"rep_index": 0, "member_indices": [0, 1], "text": "same"}], "degraded": False}

    result = propose_issues(
        [finding("semantic adapter repeat")],
        "owner/repo",
        existing_issues_fn=lambda **_kwargs: [{"title": "Different surface title", "body": "same issue", "evidence": ["scripts/reflection_loop.py:42 stale finding emitted twice"]}],
        dedup_fn=make_loci_dedup_fn(fake_loci_semantic_dedup_fn),
        **paths(files),
    )

    assert len(seen) == 1
    assert seen[0]["threshold"] == 0.88
    assert "scripts/reflection_loop.py:42 stale finding emitted twice" in seen[0]["items"][0]["text"]
    assert result["proposals"] == []
    assert result["deduped"][0]["method"].startswith("loci_semantic_dedup")
