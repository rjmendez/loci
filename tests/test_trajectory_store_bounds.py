"""Trajectory/adherence store bounds, reward integrity, scorer weights, recorder-only tool.

Slice S3 of the Tripwire-Governor plan.

Mutation runs (each applied to the production code, confirmed red, reverted):
- loci/jsonl_bounds.py: drop the rotation (``rotate = rows >= max_rows`` -> False and
  ``size + len(data) > max_bytes`` -> False) -> the 5001st-row, 1025th-KiB-row and 10001-row tests go red.
- loci/jsonl_bounds.py: ``rows >= max_rows`` -> ``rows > max_rows`` -> test_row_5001_rotates... goes red.
- loci/jsonl_bounds.py: ``size + len(data) > max_bytes`` -> ``>=`` -> the exactly-1-MiB twin goes red.
- loci/jsonl_bounds.py: tail_lines reading the whole file -> the exact bytes_read == 12288 test goes red.
- loci/drpg.py: delete the range check in _check_reward -> the r_cor=1.5 / NaN / bool tests go red.
- loci/drpg.py: delete the strict missing-key branch -> the missing-key test goes red.
- loci/drpg.py: weights_from_mapping drops the r_rem -> r_mem mapping -> the legacy-weight test goes red.
- loci/drpg.py: evolve_policy scores with TrajectoryScorer() instead of from_policy -> the evolve test goes red.
- mcp/tools/drpg_tools.py: register policy_trajectory_push again -> the exact tool-name test goes red.
"""
import importlib.util
import json
import logging
from pathlib import Path

import pytest

from loci import drpg
from loci.drpg import TrajectoryScorer, TrajectoryStore, evolve_policy
from loci.jsonl_bounds import MAX_FILE_BYTES, MAX_FILE_ROWS, append_bounded, tail_lines
from loci.policy import Policy, PolicyBackend, PolicyTier

REPO = Path(__file__).resolve().parents[1]


def _vector(**overrides) -> dict:
    vec = {"r_ret": 0.5, "r_mem": 0.0, "r_ovp": 0.0, "r_bkp": 0.0, "r_ref": 0.0, "r_step": 0.0, "r_cor": 0.0}
    vec.update(overrides)
    return vec


def _row(i: int) -> dict:
    return {"step": i, "action": "ANSWER", "reward": _vector(), "_ts": "t"}


def _lines(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fixed_size_line(i: int, size: int) -> str:
    """A JSON line (no newline) of exactly size-1 bytes, so with its newline it is `size` bytes."""
    base = json.dumps({"i": f"{i:05d}", "pad": ""})
    return json.dumps({"i": f"{i:05d}", "pad": "x" * (size - 1 - len(base))})


# ---------------------------------------------------------------------------
# rotation
# ---------------------------------------------------------------------------

def test_5000_rows_do_not_rotate(tmp_path):
    store = TrajectoryStore(tmp_path)
    for i in range(MAX_FILE_ROWS):
        store.push("pol", _row(i))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pol.jsonl"]
    assert len(_lines(tmp_path / "pol.jsonl")) == 5000


def test_row_5001_rotates_and_the_previous_file_holds_5000_rows(tmp_path):
    store = TrajectoryStore(tmp_path)
    for i in range(MAX_FILE_ROWS + 1):
        store.push("pol", _row(i))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pol.jsonl", "pol.jsonl.1"]
    rotated = _lines(tmp_path / "pol.jsonl.1")
    assert len(rotated) == 5000
    assert (rotated[0]["step"], rotated[-1]["step"]) == (0, 4999)
    assert [r["step"] for r in _lines(tmp_path / "pol.jsonl")] == [5000]
    assert store.list_policies() == ["pol"]  # the rotated file is not mistaken for another policy


def test_rotation_continues_with_increasing_suffixes(tmp_path):
    store = TrajectoryStore(tmp_path)
    for i in range(2 * MAX_FILE_ROWS + 1):
        store.push("pol", _row(i))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pol.jsonl", "pol.jsonl.1", "pol.jsonl.2"]
    assert [len(_lines(tmp_path / n)) for n in ("pol.jsonl.1", "pol.jsonl.2", "pol.jsonl")] == [5000, 5000, 1]
    assert _lines(tmp_path / "pol.jsonl.2")[0]["step"] == 5000


def test_exactly_1_mib_does_not_rotate_but_the_next_row_does(tmp_path):
    path = tmp_path / "f.jsonl"
    for i in range(1024):
        assert append_bounded(path, _fixed_size_line(i, 1024)) is False
    assert path.stat().st_size == MAX_FILE_BYTES == 1048576
    assert [p.name for p in tmp_path.iterdir()] == ["f.jsonl"]
    assert append_bounded(path, _fixed_size_line(1024, 1024)) is True
    assert (tmp_path / "f.jsonl.1").stat().st_size == 1048576
    assert len((tmp_path / "f.jsonl.1").read_text(encoding="utf-8").splitlines()) == 1024
    assert path.stat().st_size == 1024
    assert json.loads(path.read_text(encoding="utf-8"))["i"] == "01024"


def test_a_single_row_over_the_byte_limit_is_refused_and_writes_nothing(tmp_path):
    path = tmp_path / "f.jsonl"
    with pytest.raises(ValueError, match=r"row of 1048578 bytes exceeds the 1048576-byte file limit"):
        append_bounded(path, "x" * (MAX_FILE_BYTES + 1))
    assert not path.exists()
    assert append_bounded(path, "x" * (MAX_FILE_BYTES - 1)) is False  # twin: exactly 1 MiB with its newline fits
    assert path.stat().st_size == MAX_FILE_BYTES


def test_a_legacy_file_already_over_the_row_cap_is_rotated_whole_on_the_next_push(tmp_path):
    store = TrajectoryStore(tmp_path)
    legacy = [json.dumps(_row(i)) for i in range(6000)]
    (tmp_path / "pol.jsonl").write_text("\n".join(legacy) + "\n", encoding="utf-8")
    assert [s["step"] for s in store.read("pol")] == list(range(1000, 6000))  # reads are capped to the tail
    store.push("pol", _row(6000))
    assert len(_lines(tmp_path / "pol.jsonl.1")) == 6000
    assert [r["step"] for r in _lines(tmp_path / "pol.jsonl")] == [6000]


def test_the_row_count_is_recounted_when_the_file_changed_behind_the_cache(tmp_path):
    path = tmp_path / "f.jsonl"
    for i in range(3):
        append_bounded(path, json.dumps({"i": i}), max_rows=4)
    with open(path, "ab") as fh:  # another writer appends two rows behind our back
        fh.write(b'{"i": 3}\n{"i": 4}\n')
    assert append_bounded(path, json.dumps({"i": 5}), max_rows=4) is True
    assert len((tmp_path / "f.jsonl.1").read_text(encoding="utf-8").splitlines()) == 5


# ---------------------------------------------------------------------------
# reads bounded to the tail
# ---------------------------------------------------------------------------

class _CountingFile:
    """Recording wrapper: delegates to the real file and counts the bytes read() returns."""

    def __init__(self, fh, log):
        self._fh = fh
        self._log = log

    def seek(self, *args):
        return self._fh.seek(*args)

    def tell(self):
        return self._fh.tell()

    def read(self, size=-1):
        data = self._fh.read(size)
        self._log["bytes"] += len(data)
        self._log["reads"] += 1
        return data

    def __enter__(self):
        self._fh.__enter__()
        return self

    def __exit__(self, *exc):
        return self._fh.__exit__(*exc)


def _counting_opener(log):
    def opener(path, mode):
        return _CountingFile(open(path, mode), log)
    return opener


def _write_5000_lines_of_100_bytes(path: Path) -> None:
    lines = []
    for i in range(5000):
        base = json.dumps({"step": i, "pad": ""})
        lines.append(json.dumps({"step": i, "pad": "x" * (99 - len(base))}))
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))  # bytes: no newline translation on Windows
    assert path.stat().st_size == 500000


def test_read_last_100_reads_at_most_100_rows_and_three_blocks(tmp_path):
    path = tmp_path / "pol.jsonl"
    _write_5000_lines_of_100_bytes(path)
    log = {"bytes": 0, "reads": 0}
    lines = tail_lines(path, 100, opener=_counting_opener(log))
    assert [json.loads(line)["step"] for line in lines] == list(range(4900, 5000))
    # 2 blocks of 4096 B hold 81 newlines (< 101 needed); 3 blocks hold 122.
    assert (log["reads"], log["bytes"]) == (3, 12288)


def test_store_read_with_a_limit_goes_through_the_bounded_tail(tmp_path, monkeypatch):
    _write_5000_lines_of_100_bytes(tmp_path / "pol.jsonl")
    log = {"bytes": 0, "reads": 0}
    real = drpg.tail_lines
    monkeypatch.setattr(drpg, "tail_lines", lambda path, n: real(path, n, opener=_counting_opener(log)))
    steps = TrajectoryStore(tmp_path).read("pol", limit=100)
    assert [s["step"] for s in steps] == list(range(4900, 5000))
    assert log["bytes"] == 12288


def test_store_read_default_returns_the_whole_5000_row_file(tmp_path):
    _write_5000_lines_of_100_bytes(tmp_path / "pol.jsonl")
    steps = TrajectoryStore(tmp_path).read("pol")
    assert [s["step"] for s in steps] == list(range(5000))


def test_tail_lines_edge_cases(tmp_path):
    path = tmp_path / "f.jsonl"
    assert tail_lines(path, 3) == []  # missing file
    path.write_bytes(b'{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    assert tail_lines(path, 2) == [b'{"a": 2}', b'{"a": 3}']
    assert tail_lines(path, 99) == [b'{"a": 1}', b'{"a": 2}', b'{"a": 3}']  # fewer lines than asked
    assert tail_lines(path, 0) == []
    path.write_bytes(b'{"a": 1}\n{"a": 2}')  # last line lacks its newline
    assert tail_lines(path, 1) == [b'{"a": 2}']


def _load_adherence():
    spec = importlib.util.spec_from_file_location("adherence_bounds_under_test", REPO / "mcp" / "adherence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _adherence_record(adherence, i):
    return adherence.PolicyAdherenceRecord(
        policy_id="pol", call_id=f"c{i}", tier_requested="light", model_requested="a", model_used="b",
        was_downgraded=False, r_cor=0.5, r_step=0.0, timestamp="t", context={})


def test_adherence_file_rotates_at_row_5001_and_reads_only_the_tail(tmp_path):
    adherence = _load_adherence()
    store = adherence.AdherenceStore(tmp_path)
    for i in range(MAX_FILE_ROWS + 1):
        store.record(_adherence_record(adherence, i))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pol.jsonl", "pol.jsonl.1"]
    assert len((tmp_path / "pol.jsonl.1").read_text(encoding="utf-8").splitlines()) == 5000
    assert [r.call_id for r in store.read("pol", limit=100)] == ["c5000"]  # the new file holds the one newest row
    assert [r.call_id for r in store.read("missing", limit=100)] == []


def test_adherence_read_returns_the_last_n_records_in_order(tmp_path):
    adherence = _load_adherence()
    store = adherence.AdherenceStore(tmp_path)
    for i in range(12):
        store.record(_adherence_record(adherence, i))
    assert [r.call_id for r in store.read("pol", limit=5)] == ["c7", "c8", "c9", "c10", "c11"]


# ---------------------------------------------------------------------------
# reward integrity
# ---------------------------------------------------------------------------

def test_push_rejects_r_cor_above_1_and_writes_nothing(tmp_path):
    store = TrajectoryStore(tmp_path)
    with pytest.raises(ValueError, match=r"invalid reward: r_cor=1\.5 is not a finite number in \[-1, 1\]"):
        store.push("pol", {"step": 0, "reward": _vector(r_cor=1.5)})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", [1.0, -1.0, 0, 0.25])
def test_push_accepts_values_on_and_inside_the_boundary(tmp_path, value):
    TrajectoryStore(tmp_path).push("pol", {"step": 0, "reward": _vector(r_cor=value)})
    assert _lines(tmp_path / "pol.jsonl")[0]["reward"]["r_cor"] == value


@pytest.mark.parametrize("bad", [1.0000001, -1.0000001, float("nan"), float("inf"), True, "0.5", None])
def test_push_rejects_out_of_range_and_non_numeric_values(tmp_path, bad):
    with pytest.raises(ValueError, match=r"invalid reward: r_step="):
        TrajectoryStore(tmp_path).push("pol", {"step": 0, "reward": _vector(r_step=bad)})
    assert list(tmp_path.iterdir()) == []


def test_push_rejects_a_reward_that_is_not_an_object(tmp_path):
    with pytest.raises(ValueError, match=r"invalid reward: expected an object of reward values, got list"):
        TrajectoryStore(tmp_path).push("pol", {"step": 0, "reward": [1, 2]})


def test_strict_push_rejects_a_missing_required_key_and_stores_the_full_vector(tmp_path):
    store = TrajectoryStore(tmp_path)
    partial = _vector()
    del partial["r_cor"]
    with pytest.raises(ValueError, match=r"invalid reward: missing required key\(s\) r_cor"):
        store.push("pol", {"step": 0, "reward": partial}, strict=True)
    assert list(tmp_path.iterdir()) == []
    full = _vector(r_cor=1.0, r_step=-0.5)
    store.push("pol", {"step": 1, "reward": full}, strict=True)
    assert _lines(tmp_path / "pol.jsonl")[0]["reward"] == full


def test_strict_push_rejects_a_step_without_a_reward(tmp_path):
    with pytest.raises(ValueError, match=r"invalid reward: step has no reward vector"):
        TrajectoryStore(tmp_path).push("pol", {"step": 0}, strict=True)


def test_non_strict_push_still_accepts_a_partial_vector(tmp_path):
    # Legacy callers (refusal steps, older tests) push partial vectors; only range is enforced for them.
    TrajectoryStore(tmp_path).push("pol", {"step": 0, "reward": {"r_ret": 1.0}})
    assert _lines(tmp_path / "pol.jsonl")[0]["reward"] == {"r_ret": 1.0}


def test_a_rejected_push_leaves_an_existing_file_byte_for_byte_unchanged(tmp_path):
    store = TrajectoryStore(tmp_path)
    store.push("pol", _row(0))
    before = (tmp_path / "pol.jsonl").read_bytes()
    with pytest.raises(ValueError, match="invalid reward"):
        store.push("pol", {"step": 1, "reward": _vector(r_ret=2.0)})
    assert (tmp_path / "pol.jsonl").read_bytes() == before


# ---------------------------------------------------------------------------
# scorer weights
# ---------------------------------------------------------------------------

def test_scorer_applies_a_mapping_weight_exactly():
    assert TrajectoryScorer({"r_mem": -0.5}).score([{"reward": {"r_mem": 1.0}}]) == -0.5


def test_scorer_from_policy_uses_get_reward_vector_weights(monkeypatch, caplog):
    policy = Policy(policy_id="pol", title="t", tier=PolicyTier.STANDARD)
    monkeypatch.setattr(policy, "get_reward_vector_weights", lambda: {"r_mem": -0.5})
    with caplog.at_level(logging.WARNING, logger="loci.drpg"):
        assert TrajectoryScorer.from_policy(policy).score([{"reward": {"r_mem": 1.0}}]) == -0.5
    assert caplog.records == []  # the current key name needs no mapping and logs nothing


def test_legacy_r_rem_weight_is_mapped_to_r_mem_with_one_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="loci.drpg"):
        scorer = TrajectoryScorer({"r_rem": -0.5})
    assert scorer.weights == (0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert scorer.score([{"reward": {"r_mem": 1.0}}]) == -0.5
    assert [r.getMessage() for r in caplog.records] == [
        "reward weight key 'r_rem' is the legacy name of 'r_mem'; mapped (use 'r_mem')"]


def test_explicit_r_mem_wins_over_the_legacy_key_and_the_legacy_key_is_still_reported(caplog):
    with caplog.at_level(logging.WARNING, logger="loci.drpg"):
        scorer = TrajectoryScorer({"r_rem": 9.0, "r_mem": -0.5})
    assert scorer.weights[1] == -0.5
    assert len(caplog.records) == 1


def test_unknown_weight_key_is_an_error():
    with pytest.raises(ValueError, match=r"unknown reward weight key 'r_zzz'"):
        TrajectoryScorer({"r_zzz": 1.0})


def test_standard_policy_weights_score_a_known_step_exactly():
    policy = Policy(policy_id="pol", title="t", tier=PolicyTier.STANDARD)  # discovery: r_ret 1, r_ref -0.5, r_cor 0.5
    step = {"reward": {"r_ret": 0.5, "r_ref": 1.0, "r_cor": 1.0}}
    assert TrajectoryScorer.from_policy(policy).score([step]) == 0.5  # 0.5 - 0.5 + 0.5
    assert TrajectoryScorer().score([step]) == 2.5  # the fixed default tuple: 0.5 + 1.0 + 1.0


def test_evolve_policy_scores_with_the_policys_own_weights(tmp_path):
    backend = PolicyBackend(tmp_path / "policies")
    store = TrajectoryStore(tmp_path / "traj")
    backend.store(Policy(policy_id="pol", title="t", tier=PolicyTier.STANDARD))
    for i in range(3):
        store.push("pol", {"step": i, "action": "ANSWER", "reward": _vector(r_ret=0.5, r_ref=1.0, r_cor=1.0)})
    result = evolve_policy("pol", backend, store)
    # policy weights give 0.5 -> keep; the old fixed tuple would give 2.5 -> promote
    assert (result["action"], result["score"], result["steps_analyzed"]) == ("keep", 0.5, 3)


# ---------------------------------------------------------------------------
# recorder-only reward writes
# ---------------------------------------------------------------------------

def _tools_module():
    spec = importlib.util.spec_from_file_location("drpg_tools_bounds_under_test", REPO / "mcp" / "tools" / "drpg_tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_trajectory_push_is_not_an_agent_visible_tool():
    names = []

    class Server:
        def tool(self):
            def deco(fn):
                names.append(fn.__name__)
                return fn
            return deco

    _tools_module().register(Server())
    assert sorted(names) == ["policy_evolve", "policy_evolve_all"]


def test_the_recorder_entry_point_stores_the_same_vector(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCI_HOME", str(tmp_path / "loci-home"))
    monkeypatch.delenv("LOCI_TRAJECTORY_DIR", raising=False)
    vector = _vector(r_cor=1.0, r_step=-0.5)
    out = _tools_module().record_trajectory_step("pol-r", 3, "SEARCH", vector)
    assert out == {"pushed": True, "policy_id": "pol-r", "step": 3}
    row = _lines(tmp_path / "loci-home" / "trajectories" / "pol-r.jsonl")[0]
    assert (row["step"], row["action"], row["reward"], row["policy_id"]) == (3, "SEARCH", vector, "pol-r")


def test_the_recorder_entry_point_refuses_an_incomplete_or_out_of_range_vector(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCI_HOME", str(tmp_path / "loci-home"))
    monkeypatch.delenv("LOCI_TRAJECTORY_DIR", raising=False)
    recorder = _tools_module().record_trajectory_step
    with pytest.raises(ValueError, match=r"invalid reward: missing required key\(s\) r_mem"):
        recorder("pol-r", 0, "SEARCH", {k: 0.0 for k in _vector() if k != "r_mem"})
    with pytest.raises(ValueError, match=r"invalid reward: r_cor=1\.5"):
        recorder("pol-r", 0, "SEARCH", _vector(r_cor=1.5))
    assert not (tmp_path / "loci-home" / "trajectories" / "pol-r.jsonl").exists()
