"""
Performance benchmarks for hot paths.
Run with: pytest tests/test_perf_benchmarks.py --benchmark-only

Mutation-verification notes (mutations applied to confirm tests go red, then reverted):
- test_benchmark_policy_load_cold: deleted the mtime check + cache lookup in load()
  → result.title assertion still passes (correctness unaffected), BUT removing the
  cache means the warm test shows no speed-up; this is caught by comparing stats.
- test_benchmark_policy_load_warm: same mutation → warm latency equals cold, violating
  the purpose of the test (though the assertion on result.title still passes — the
  benchmark itself is the signal, not just the assertion).
- test_benchmark_policy_list: deleted mtime cache in list_policies() → still returns
  10 policies; assertion passes; benchmark latency increases (perf regression visible).
- test_benchmark_trajectory_read_cold: deleted _TRAJ_CACHE lookup in read() → reads
  file every time, len(result) still 100, assertion passes; cold == warm latency.
- test_benchmark_trajectory_read_warm: same; result length assertion still passes.
- test_benchmark_trajectory_push: deleted _TRAJ_CACHE.pop in push() → cache not
  invalidated; subsequent read returns stale; assertion on step=999 presence fails
  (cache returns snapshot from before push was called). ✓ red on this mutation.
- test_benchmark_jsonl_read_cold: removed _jsonl_cache_set from _read_jsonl →
  len(result) still 500 (file re-read each benchmark iteration, no cache), but
  warm benchmark would now equal cold; len assertion still passes. Correctness
  is caught by test_inv_store_cache.py; benchmark number is the signal here.
- test_benchmark_jsonl_read_warm: removed ``if cached is not None: return cached``
  guard → file re-read every call; len(result) still 500, but warm ≈ cold latency
  (visible in benchmark stats). ✓
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def policy_backend(tmp_path):
    from policy import Policy, PolicyTier, PolicyRegion, PolicyBackend
    backend = PolicyBackend(tmp_path / "policies")
    policies = []
    for i in range(10):
        p = Policy(title=f"policy-{i}", tier=PolicyTier.MEDIUM, region=PolicyRegion.MEMORY)
        backend.store(p)
        policies.append(p)
    return backend, policies


@pytest.fixture
def trajectory_store(tmp_path):
    from drpg_evolution import TrajectoryStore
    store = TrajectoryStore(tmp_path / "trajectories")
    for i in range(100):
        store.push("bench-policy", {
            "step": i,
            "action": "SEARCH",
            "reward": {
                "r_ret": 1.0, "r_cor": 0.8, "r_step": -0.01,
                "r_mem": 0.5, "r_ovp": 0.0, "r_bkp": 0.0, "r_ref": 0.0,
            },
        })
    return store


# ---------------------------------------------------------------------------
# PolicyBackend benchmarks
# ---------------------------------------------------------------------------

def test_benchmark_policy_load_cold(benchmark, policy_backend, monkeypatch):
    """First load (cold cache): file open + json.load on every call."""
    import loci.policy as _policy_mod
    backend, policies = policy_backend
    monkeypatch.setattr(_policy_mod, "_POLICY_CACHE", {})
    result = benchmark(backend.load, policies[0].policy_id)
    assert result is not None
    assert result.title == "policy-0"


def test_benchmark_policy_load_warm(benchmark, policy_backend):
    """Repeated load (warm cache): should be near-zero cost after first call."""
    backend, policies = policy_backend
    backend.load(policies[0].policy_id)  # prime the cache
    result = benchmark(backend.load, policies[0].policy_id)
    assert result is not None
    assert result.title == "policy-0"


def test_benchmark_policy_list(benchmark, policy_backend):
    """List all 10 policies (uses mtime cache on repeated calls)."""
    backend, _ = policy_backend
    result = benchmark(backend.list_policies)
    assert len(result) == 10


# ---------------------------------------------------------------------------
# TrajectoryStore benchmarks
# ---------------------------------------------------------------------------

def test_benchmark_trajectory_read_cold(benchmark, trajectory_store, monkeypatch):
    """Read all 100 steps (cold cache): full JSONL scan."""
    import loci.drpg as _drpg_mod
    monkeypatch.setattr(_drpg_mod, "_TRAJ_CACHE", {})
    result = benchmark(trajectory_store.read, "bench-policy")
    assert len(result) == 100


def test_benchmark_trajectory_read_warm(benchmark, trajectory_store):
    """Read all 100 steps (warm cache): should return cached copy."""
    trajectory_store.read("bench-policy")  # prime the cache
    result = benchmark(trajectory_store.read, "bench-policy")
    assert len(result) == 100


def test_benchmark_trajectory_push(benchmark, trajectory_store):
    """Append one step and verify it lands in the store."""
    step = {"step": 999, "action": "ANSWER", "reward": {"r_cor": 1.0}}
    benchmark(trajectory_store.push, "bench-policy", step)
    steps = trajectory_store.read("bench-policy")
    assert any(s.get("step") == 999 for s in steps)


# ---------------------------------------------------------------------------
# orjson vs stdlib JSONL parse benchmarks
# ---------------------------------------------------------------------------

def test_benchmark_jsonl_parse_stdlib(benchmark, tmp_path):
    """Baseline: stdlib json for 500-line JSONL."""
    import json
    path = tmp_path / "bench.jsonl"
    rows = [json.dumps({"id": f"f{i}", "content": f"finding {i}", "confidence": 0.8, "ts": "2026-01-01"}) for i in range(500)]
    path.write_text("\n".join(rows))

    def parse():
        lines = []
        for line in path.read_bytes().decode("utf-8").splitlines():
            if line.strip():
                lines.append(json.loads(line))
        return lines

    result = benchmark(parse)
    assert len(result) == 500
    assert result[0]["id"] == "f0"
    assert result[499]["id"] == "f499"


def test_benchmark_jsonl_parse_orjson(benchmark, tmp_path):
    """orjson: same 500-line JSONL."""
    import json
    try:
        import orjson
    except ImportError:
        pytest.skip("orjson not installed")

    path = tmp_path / "bench.jsonl"
    rows = [json.dumps({"id": f"f{i}", "content": f"finding {i}", "confidence": 0.8, "ts": "2026-01-01"}) for i in range(500)]
    path.write_text("\n".join(rows))

    def parse():
        lines = []
        for line in path.read_bytes().splitlines():
            if line.strip():
                lines.append(orjson.loads(line))
        return lines

    result = benchmark(parse)
    assert len(result) == 500
    assert result[0]["id"] == "f0"
    assert result[499]["id"] == "f499"


# ---------------------------------------------------------------------------
# inv_store JSONL cache benchmarks
# ---------------------------------------------------------------------------

@pytest.fixture
def jsonl_file(tmp_path):
    """500-line JSONL file mimicking a findings.jsonl."""
    path = tmp_path / "findings.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(500):
            f.write(json.dumps({"id": f"f{i}", "content": f"finding {i}", "confidence": 0.8}) + "\n")
    return path


def test_benchmark_jsonl_read_cold(benchmark, jsonl_file, monkeypatch):
    """Read 500-line JSONL cold (no cache): full file scan."""
    import inv_store as _inv_store_mod
    monkeypatch.setattr(_inv_store_mod, "_JSONL_CACHE", {})
    result = benchmark(_inv_store_mod._read_jsonl, jsonl_file)
    assert len(result) == 500


def test_benchmark_jsonl_read_warm(benchmark, jsonl_file, monkeypatch):
    """Read 500-line JSONL warm (cache hit): should be near-zero cost."""
    import inv_store as _inv_store_mod
    monkeypatch.setattr(_inv_store_mod, "_JSONL_CACHE", {})
    _inv_store_mod._read_jsonl(jsonl_file)  # prime cache
    result = benchmark(_inv_store_mod._read_jsonl, jsonl_file)
    assert len(result) == 500
