"""instrumentation_log: bounded, append-only, fail-open JSONL."""
import json

import pytest

import instrumentation_log as IL


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_append_writes_one_sorted_json_line_per_row(tmp_path):
    path = tmp_path / "log.jsonl"
    assert IL.append_rows(path, [{"b": 2, "a": 1}, {"c": 3}]) is True
    assert path.read_text() == '{"a": 1, "b": 2}\n{"c": 3}\n'


def test_empty_rows_write_nothing(tmp_path):
    path = tmp_path / "log.jsonl"
    assert IL.append_rows(path, []) is True
    assert not path.exists()


def test_rotation_keeps_newest_rows_live_and_drops_past_keep(tmp_path):
    path = tmp_path / "log.jsonl"
    line_len = len(json.dumps({"i": 0}) + "\n")  # 9 bytes
    # Cap fits exactly two rows per generation.
    for i in range(7):
        assert IL.append_rows(path, [{"i": i}], limit_bytes=2 * line_len, keep=2)
    assert _rows(path) == [{"i": 6}]
    assert _rows(tmp_path / "log.jsonl.1") == [{"i": 4}, {"i": 5}]
    assert _rows(tmp_path / "log.jsonl.2") == [{"i": 2}, {"i": 3}]
    # Generation 3 would exceed keep=2: rows 0 and 1 are gone, nothing else is.
    assert not (tmp_path / "log.jsonl.3").exists()


def test_a_single_call_is_never_split_across_generations(tmp_path):
    path = tmp_path / "log.jsonl"
    IL.append_rows(path, [{"i": 0}], limit_bytes=20, keep=3)
    IL.append_rows(path, [{"i": 1}, {"i": 2}, {"i": 3}], limit_bytes=20, keep=3)
    assert _rows(path) == [{"i": 1}, {"i": 2}, {"i": 3}]
    assert _rows(tmp_path / "log.jsonl.1") == [{"i": 0}]


def test_under_the_cap_nothing_rotates(tmp_path):
    path = tmp_path / "log.jsonl"
    for i in range(3):
        IL.append_rows(path, [{"i": i}], limit_bytes=10_000, keep=3)
    assert _rows(path) == [{"i": 0}, {"i": 1}, {"i": 2}]
    assert not (tmp_path / "log.jsonl.1").exists()


def test_failure_returns_false_and_never_raises(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    assert IL.append_rows(blocker / "log.jsonl", [{"a": 1}]) is False


def test_unserialisable_row_fails_open(tmp_path):
    path = tmp_path / "log.jsonl"
    assert IL.append_rows(path, [{"a": object()}]) is False
    assert not path.exists()


@pytest.mark.parametrize("raw,default,expected", [
    (None, True, True),
    ("", True, True),
    ("  ", False, False),
    ("0", True, False),
    ("false", True, False),
    ("OFF", True, False),
    ("no", True, False),
    ("1", False, True),
    ("yes", False, True),
])
def test_env_enabled(monkeypatch, raw, default, expected):
    if raw is None:
        monkeypatch.delenv("LOCI_TEST_FLAG", raising=False)
    else:
        monkeypatch.setenv("LOCI_TEST_FLAG", raw)
    assert IL.env_enabled("LOCI_TEST_FLAG", default) is expected


def test_size_and_keep_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("LOCI_INSTRUMENTATION_LOG_MAX_BYTES", raising=False)
    monkeypatch.delenv("LOCI_INSTRUMENTATION_LOG_KEEP", raising=False)
    assert IL.max_bytes() == 4 * 1024 * 1024
    assert IL.keep_generations() == 3
    monkeypatch.setenv("LOCI_INSTRUMENTATION_LOG_MAX_BYTES", "100")
    monkeypatch.setenv("LOCI_INSTRUMENTATION_LOG_KEEP", "0")
    assert IL.max_bytes() == 4096  # clamped: a tiny cap would rotate on every write
    assert IL.keep_generations() == 1
    monkeypatch.setenv("LOCI_INSTRUMENTATION_LOG_MAX_BYTES", "not-a-number")
    assert IL.max_bytes() == 4 * 1024 * 1024
