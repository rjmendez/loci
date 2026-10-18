"""policy_evolve_all repair, evolve_all_policies bounds, policy_id validation, store hermeticity.

Slice S2 of the Tripwire-Governor plan.  Every store here lives under tmp_path: LOCI_HOME is
pointed there with monkeypatch, so the default-constructed PolicyBackend() and
_default_trajectory_store() that the MCP wrapper uses land in the temp dir.

Mutation runs (each applied to the production code, confirmed red, reverted):
- mcp/tools/drpg_tools.py: use ``backend.base_dir`` again -> test_wrapper_evolves_three_policies
  and test_wrapper_zero_rows_is_status_1 go red (AttributeError swallowed into an error result).
- loci/drpg.py: evolve_all_policies back to ``ProcessPoolExecutor(max_workers=...)`` ->
  test_evolve_all_never_constructs_a_process_pool goes red.
- loci/policy.py: allow '/' in POLICY_ID_PATTERN -> test_validate_policy_id_rejects_unsafe_names[a/b]
  goes red (PolicyBackend._policy_file's containment check is a second layer, so the store-level
  test alone cannot see this one; the containment check has its own tests, bypassing the pattern).
- loci/policy.py: delete the ``path.parent != base`` check -> the containment tests go red.
- loci/drpg.py: drop the ``> MAX_POLICIES_PER_CALL`` check -> the limit tests go red.
- testsupport/loci_hermetic.py: drop LOCI_HOME from install() -> test_default_stores_resolve_inside_the_isolated_root goes red.
"""
import asyncio
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from loci import drpg
from loci.drpg import MAX_POLICIES_PER_CALL, PolicyLimitError, TrajectoryStore, evolve_all_policies
from loci.policy import Policy, PolicyBackend, PolicyTier, loci_home, validate_policy_id

REPO = Path(__file__).resolve().parents[1]


def _full_vector(r_ret: float = 0.5) -> dict:
    return {"r_ret": r_ret, "r_mem": 0.0, "r_ovp": 0.0, "r_bkp": 0.0,
            "r_ref": 0.0, "r_step": 0.0, "r_cor": 0.0}


def _row(i: int) -> dict:
    return {"step": i, "action": "ANSWER", "reward": _full_vector()}


def _register_tools():
    """The registered DRPG tools, by name, from a fake server that records what is registered."""
    spec = importlib.util.spec_from_file_location("drpg_tools_under_test", REPO / "mcp" / "tools" / "drpg_tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Server:
        def __init__(self):
            self.tools = {}

        def tool(self):
            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn
            return deco

    server = Server()
    module.register(server)
    return server.tools


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCI_HOME", str(tmp_path / "loci-home"))
    monkeypatch.delenv("LOCI_TRAJECTORY_DIR", raising=False)
    return tmp_path / "loci-home"


def _seed(home_dir: Path, rows_per_policy: dict) -> None:
    backend = PolicyBackend(home_dir / "policies")
    store = TrajectoryStore(home_dir / "trajectories")
    for pid, rows in rows_per_policy.items():
        backend.store(Policy(policy_id=pid, title=f"policy {pid}", tier=PolicyTier.STANDARD))
        for i in range(rows):
            store.push(pid, _row(i))


# ---------------------------------------------------------------------------
# the MCP wrapper
# ---------------------------------------------------------------------------

def test_wrapper_evolves_three_policies(home):
    _seed(home, {"pol-a": 2, "pol-b": 3, "pol-c": 4})
    out = json.loads(_register_tools()["policy_evolve_all"]())
    expected = [
        {"policy_id": pid, "action": "keep", "score": 0.5, "suggestions": [], "steps_analyzed": n}
        for pid, n in (("pol-a", 2), ("pol-b", 3), ("pol-c", 4))
    ]
    assert out == {
        "total": 3, "promoted": 0, "demoted": 0, "errors": 0,
        "policies_evolved": 3, "rows_read": 9, "results": expected, "status": 0,
    }


def test_wrapper_zero_rows_is_status_1(home):
    _seed(home, {"pol-a": 0})
    out = json.loads(_register_tools()["policy_evolve_all"](["pol-a"]))
    assert out["rows_read"] == 0
    assert out["status"] == 1
    assert out["error"] == "rows_read=0"
    # the policy itself was evolved (a policy with no rows scores 0.0 and is demoted)
    assert (out["total"], out["demoted"], out["policies_evolved"]) == (1, 1, 1)


def test_wrapper_refuses_more_than_64_policies(home):
    ids = [f"p{i:03d}" for i in range(MAX_POLICIES_PER_CALL + 1)]
    out = json.loads(_register_tools()["policy_evolve_all"](ids))
    assert out == {"error": "65 policies requested; at most 64 per call", "status": 1}


def test_wrapper_accepts_exactly_64_policies(home):
    _seed(home, {f"p{i:03d}": 1 for i in range(MAX_POLICIES_PER_CALL)})
    out = json.loads(_register_tools()["policy_evolve_all"]())
    assert (out["status"], out["policies_evolved"], out["rows_read"]) == (0, 64, 64)


# ---------------------------------------------------------------------------
# evolve_all_policies: serial, bounded
# ---------------------------------------------------------------------------

class _RecordingPool(ThreadPoolExecutor):
    """Stand-in for ProcessPoolExecutor that records every construction (and still works)."""
    constructed = []

    def __init__(self, *args, **kwargs):
        type(self).constructed.append((args, kwargs))
        super().__init__(*args, **kwargs)


@pytest.fixture
def pool_recorder(monkeypatch):
    import concurrent.futures
    import concurrent.futures.process

    _RecordingPool.constructed = []
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _RecordingPool)
    monkeypatch.setattr(concurrent.futures.process, "ProcessPoolExecutor", _RecordingPool)
    monkeypatch.setattr(drpg, "ProcessPoolExecutor", _RecordingPool, raising=False)
    return _RecordingPool


@pytest.fixture
def evolve_calls(monkeypatch):
    calls = []

    def evolve_policy(policy_id, backend, store):
        calls.append(policy_id)
        return {"policy_id": policy_id, "action": "keep", "score": 0.5, "suggestions": [], "steps_analyzed": 1}

    monkeypatch.setattr(drpg, "evolve_policy", evolve_policy)
    return calls


def test_evolve_all_never_constructs_a_process_pool(tmp_path, pool_recorder, evolve_calls):
    ids = [f"p{i:03d}" for i in range(MAX_POLICIES_PER_CALL)][::-1]  # given in reverse order
    results = asyncio.run(evolve_all_policies(ids, tmp_path / "pol", tmp_path / "traj", max_workers=8))
    assert pool_recorder.constructed == []
    assert evolve_calls == sorted(ids)
    assert [r["policy_id"] for r in results] == sorted(ids)
    assert len(results) == 64


def test_evolve_all_policy_65_is_refused_before_any_work(tmp_path, pool_recorder, evolve_calls):
    ids = [f"p{i:03d}" for i in range(MAX_POLICIES_PER_CALL + 1)]
    with pytest.raises(PolicyLimitError, match=r"65 policies requested; at most 64 per call"):
        asyncio.run(evolve_all_policies(ids, tmp_path / "pol", tmp_path / "traj"))
    assert evolve_calls == []
    assert pool_recorder.constructed == []


def test_evolve_all_evolves_duplicates_once(tmp_path, evolve_calls):
    asyncio.run(evolve_all_policies(["b", "a", "b"], tmp_path / "pol", tmp_path / "traj"))
    assert evolve_calls == ["a", "b"]


def test_evolve_all_empty_is_empty(tmp_path):
    assert asyncio.run(evolve_all_policies([], tmp_path / "pol", tmp_path / "traj")) == []


def test_evolve_all_traversal_id_is_an_error_result_and_writes_nothing(tmp_path):
    results = asyncio.run(evolve_all_policies(["../x"], tmp_path / "pol", tmp_path / "traj"))
    assert len(results) == 1
    assert results[0]["policy_id"] == "../x"
    assert results[0]["action"] == "error"
    assert "invalid policy_id" in results[0]["error"]
    assert not (tmp_path / "x.jsonl").exists() and not (tmp_path / "x.json").exists()


# ---------------------------------------------------------------------------
# policy_id validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["../x", "a/b", "a\\b", "C:evil", "", "a" * 65, "Upper", ".hidden", "a b"])
def test_invalid_policy_ids_are_refused(tmp_path, bad):
    backend = PolicyBackend(tmp_path / "policies")
    with pytest.raises(ValueError, match="invalid policy_id"):
        backend.store(Policy(policy_id=bad, title="t"))
    with pytest.raises(ValueError, match="invalid policy_id"):
        backend.load(bad)
    assert list(tmp_path.rglob("*.json")) == []


@pytest.mark.parametrize("bad", ["../x", "a/b", "a\\b", "C:evil", "", "a" * 65, "Upper", ".hidden", "a b", "a\n"])
def test_validate_policy_id_rejects_unsafe_names(bad):
    # The pattern is the first layer; PolicyBackend._policy_file's containment check is the second.
    with pytest.raises(ValueError, match="invalid policy_id"):
        validate_policy_id(bad)


@pytest.mark.parametrize("good", ["a", "0", "tier-nano.v2", "a_b", "a" * 64, "12333001-2bbe-496f-815a-5e837551dedf"])
def test_validate_policy_id_returns_a_safe_name_unchanged(good):
    assert validate_policy_id(good) == good


@pytest.mark.parametrize("escaping", ["../x", "a/b", "sub/../../x"])
def test_containment_check_refuses_a_path_that_leaves_the_policy_dir_even_if_the_pattern_is_bypassed(
        tmp_path, monkeypatch, escaping):
    import loci.policy as policy_module
    monkeypatch.setattr(policy_module, "validate_policy_id", lambda policy_id: policy_id)
    backend = PolicyBackend(tmp_path / "policies")
    with pytest.raises(ValueError, match=r"invalid policy_id: .*\(resolves outside the policy directory"):
        backend.load(escaping)


def test_containment_check_accepts_a_plain_name_when_the_pattern_is_bypassed(tmp_path, monkeypatch):
    import loci.policy as policy_module
    monkeypatch.setattr(policy_module, "validate_policy_id", lambda policy_id: policy_id)
    assert PolicyBackend(tmp_path / "policies").load("plain") is None


def test_valid_policy_id_is_stored_at_exactly_its_name_inside_the_base_dir(tmp_path):
    backend = PolicyBackend(tmp_path / "policies")
    result = backend.store(Policy(policy_id="tier-nano.v2", title="t", tier=PolicyTier.NANO))
    stored = tmp_path / "policies" / "tier-nano.v2.json"
    assert result == {"policy_id": "tier-nano.v2", "status": "stored"}
    assert stored.is_file()
    assert stored.resolve().parent == (tmp_path / "policies").resolve()
    assert backend.load("tier-nano.v2").policy_id == "tier-nano.v2"
    assert [p.name for p in (tmp_path / "policies").iterdir()] == ["tier-nano.v2.json"]


def test_policy_id_length_boundary_is_64(tmp_path):
    backend = PolicyBackend(tmp_path / "policies")
    assert backend.store(Policy(policy_id="a" * 64, title="t"))["status"] == "stored"
    with pytest.raises(ValueError, match="invalid policy_id"):
        backend.store(Policy(policy_id="a" * 65, title="t"))


def test_trajectory_store_refuses_a_traversal_id_and_lists_only_valid_ids(tmp_path):
    store = TrajectoryStore(tmp_path / "traj")
    with pytest.raises(ValueError, match="invalid policy_id"):
        store.push("../x", {"step": 0})
    store.push("good-1", {"step": 0})
    (tmp_path / "traj" / "Bad Name.jsonl").write_text("{}\n", encoding="utf-8")
    assert store.list_policies() == ["good-1"]
    assert not (tmp_path / "x.jsonl").exists()


# ---------------------------------------------------------------------------
# hermeticity: a bare pytest run cannot reach ~/.loci
# ---------------------------------------------------------------------------

def _hermetic():
    module = sys.modules.get("loci_hermetic")
    assert module is not None and module.installed(), "tests/conftest.py did not install loci_hermetic"
    return module


def test_default_stores_resolve_inside_the_isolated_root():
    hermetic = _hermetic()
    root = Path(hermetic.root())
    assert os.environ["LOCI_HOME"] == str(root / "loci-home")
    assert loci_home() == root / "loci-home"
    assert PolicyBackend().policy_dir == root / "loci-home" / "policies"
    assert not hermetic.is_forbidden(PolicyBackend().policy_dir)


def test_default_adherence_and_trajectory_dirs_follow_loci_home():
    spec = importlib.util.spec_from_file_location("adherence_under_test", REPO / "mcp" / "adherence.py")
    adherence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adherence)
    root = Path(_hermetic().root())
    assert adherence.AdherenceStore().adherence_dir == root / "loci-home" / "adherence"
    assert drpg._default_trajectory_store().base_dir == root / "loci-home" / "trajectories"


def test_live_policy_dir_is_refused_and_recorded():
    hermetic = _hermetic()
    live = [r for r in hermetic.forbidden_roots() if Path(r).name == ".loci"]
    assert live, "the real ~/.loci is not among the guarded roots"
    live_policies = Path(live[0]) / "policies"
    assert hermetic.is_forbidden(live_policies / "x.json")
    start = hermetic.violation_count()
    with pytest.raises(hermetic.LiveStoreAccessError, match="reaches a live Loci store"):
        PolicyBackend(live_policies)
    recorded = hermetic.take_violations(start)  # consumed here so the autouse guard does not fail this test
    assert len(recorded) == 1 and recorded[0].startswith("os.mkdir(")


def test_isolated_policy_dir_is_not_refused():
    hermetic = _hermetic()
    start = hermetic.violation_count()
    backend = PolicyBackend(Path(hermetic.root()) / "loci-home" / "policies-probe")
    assert backend.policy_dir.is_dir()
    assert hermetic.violation_count() == start
