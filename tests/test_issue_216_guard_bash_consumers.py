from pathlib import Path


def test_issue_216_guard_bash_references_removed_from_production_consumers():
    repo = Path(__file__).resolve().parents[1]
    for rel in [
        "scripts/score_trace_collector.py",
        "scripts/skill_annotation_updater.py",
        "mlops/finetune/collect.py",
        "mlops/memory/live_evo.py",
        "mlops/loop.py",
    ]:
        assert "guard_bash" not in (repo / rel).read_text(encoding="utf-8")
