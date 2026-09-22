from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_rollout_doc_covers_benchmark_rollout_and_audit_requirements():
    text = (ROOT / 'docs' / 'IMPLEMENTATION_VERIFICATION_AND_ROLLOUT.md').read_text(encoding='utf-8')
    for needle in [
        'Benchmark harness spec',
        'Phased rollout plan',
        'Acceptance and verification checks',
        'Audit-trace architecture',
        'scripts/bench_model_catalog_quality.py',
        'scripts/assign_models_from_benchmark.py',
    ]:
        assert needle in text


def test_operations_doc_links_the_rollout_guide():
    text = (ROOT / 'docs' / 'OPERATIONS.md').read_text(encoding='utf-8')
    assert 'IMPLEMENTATION_VERIFICATION_AND_ROLLOUT.md' in text
    assert 'Implementation verification and phased rollout' in text
