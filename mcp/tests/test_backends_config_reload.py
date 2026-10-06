"""backends.toml is re-read when it changes, so every Loci process sees the same config.

2026-10-06: the config was parsed once per process. After a model was pinned to a GPU in backends.toml the MCP server,
the CLI tools and the scheduled tasks disagreed about whether Ollama requests carried ``main_gpu``, and Ollama reloads
a loaded model whenever two callers disagree on it: about 15 s per call, in pairs. A pin has to reach a running
process without a restart.
"""
import os
import sys
import tomllib
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import backends as B  # noqa: E402
import model_pool  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    B._config.cache_clear()
    monkeypatch.setattr(B, "_CONFIG_RECHECK_S", 0.0)       # stat on every call unless a test says otherwise
    model_pool.clear_cache()
    yield
    B._config.cache_clear()
    model_pool.clear_cache()


def _write(path, text, mtime_ns=None):
    path.write_text(text)
    if mtime_ns is not None:
        os.utime(path, ns=(mtime_ns, mtime_ns))


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    p = tmp_path / "backends.toml"
    monkeypatch.setattr(B, "_CONFIG_PATH", str(p))
    return p


class TestReload:
    def test_an_unchanged_file_is_parsed_once(self, cfg, monkeypatch):
        _write(cfg, '[ollama]\nurl = "http://a:1"\n')
        calls = []
        real = tomllib.loads
        monkeypatch.setattr(tomllib, "loads", lambda text: (calls.append(1), real(text))[1])
        for _ in range(5):
            assert B._config() == {"ollama": {"url": "http://a:1"}}
        assert len(calls) == 1

    def test_the_same_dict_comes_back_while_nothing_changed(self, cfg):
        _write(cfg, '[ollama]\nurl = "http://a:1"\n')
        assert B._config() is B._config()

    def test_an_edit_is_seen_without_a_restart(self, cfg):
        _write(cfg, '[ollama]\nurl = "http://a:1"\n', mtime_ns=1_000_000_000)
        assert B._cfg("ollama", "url") == "http://a:1"
        _write(cfg, '[ollama]\nurl = "http://b:2"\n', mtime_ns=2_000_000_000)
        assert B._cfg("ollama", "url") == "http://b:2"

    def test_a_change_that_keeps_the_mtime_but_not_the_size_is_seen(self, cfg):
        _write(cfg, '[ollama]\nurl = "http://a:1"\n', mtime_ns=1_000_000_000)
        assert B._cfg("ollama", "url") == "http://a:1"
        _write(cfg, '[ollama]\nurl = "http://much-longer-host:1"\n', mtime_ns=1_000_000_000)
        assert B._cfg("ollama", "url") == "http://much-longer-host:1"

    def test_a_missing_file_is_empty_and_a_file_that_appears_is_read(self, cfg):
        assert B._config() == {}
        _write(cfg, '[ollama]\nurl = "http://a:1"\n')
        assert B._config() == {"ollama": {"url": "http://a:1"}}

    def test_a_file_that_disappears_goes_back_to_empty(self, cfg):
        _write(cfg, '[ollama]\nurl = "http://a:1"\n')
        assert B._config() != {}
        cfg.unlink()
        assert B._config() == {}

    def test_a_broken_file_is_empty_and_not_reparsed_until_it_changes(self, cfg, monkeypatch):
        _write(cfg, "this is [not toml", mtime_ns=1_000_000_000)
        calls = []
        real = tomllib.loads
        monkeypatch.setattr(tomllib, "loads", lambda text: (calls.append(1), real(text))[1])
        assert B._config() == {} and B._config() == {} and B._config() == {}
        assert len(calls) == 1
        _write(cfg, '[ollama]\nurl = "http://fixed:1"\n', mtime_ns=2_000_000_000)
        assert B._cfg("ollama", "url") == "http://fixed:1"

    def test_a_different_path_is_read_afresh(self, cfg, tmp_path, monkeypatch):
        _write(cfg, '[ollama]\nurl = "http://a:1"\n', mtime_ns=1_000_000_000)
        assert B._cfg("ollama", "url") == "http://a:1"
        other = tmp_path / "other.toml"
        _write(other, '[ollama]\nurl = "http://other:1"\n', mtime_ns=1_000_000_000)
        monkeypatch.setattr(B, "_CONFIG_PATH", str(other))
        assert B._cfg("ollama", "url") == "http://other:1"

    def test_two_files_with_the_same_size_and_mtime_are_not_confused(self, cfg, tmp_path, monkeypatch):
        _write(cfg, '[ollama]\nurl = "http://a:1"\n', mtime_ns=1_000_000_000)
        other = tmp_path / "other.toml"
        _write(other, '[ollama]\nurl = "http://b:1"\n', mtime_ns=1_000_000_000)      # same length, same mtime
        assert B._cfg("ollama", "url") == "http://a:1"
        monkeypatch.setattr(B, "_CONFIG_PATH", str(other))
        assert B._cfg("ollama", "url") == "http://b:1"

    def test_cache_clear_forces_a_reread(self, cfg, monkeypatch):
        _write(cfg, '[ollama]\nurl = "http://a:1"\n')
        calls = []
        real = tomllib.loads
        monkeypatch.setattr(tomllib, "loads", lambda text: (calls.append(1), real(text))[1])
        B._config()
        B._config()
        B._config.cache_clear()
        B._config()
        assert len(calls) == 2

    def test_the_test_hook_still_clears_it(self, cfg):
        _write(cfg, '[ollama]\nurl = "http://a:1"\n')
        B._config()
        B._reset_cache()
        assert B._CONFIG_CACHE["path"] is None


class TestRecheckInterval:
    def test_the_file_is_not_stat_ed_more_often_than_the_interval(self, cfg, monkeypatch):
        monkeypatch.setattr(B, "_CONFIG_RECHECK_S", 2.0)
        clock = [100.0]
        monkeypatch.setattr(B.time, "monotonic", lambda: clock[0])
        _write(cfg, '[ollama]\nurl = "http://a:1"\n', mtime_ns=1_000_000_000)
        assert B._cfg("ollama", "url") == "http://a:1"
        _write(cfg, '[ollama]\nurl = "http://b:2"\n', mtime_ns=2_000_000_000)
        clock[0] = 101.9
        assert B._cfg("ollama", "url") == "http://a:1"             # inside the interval: not even looked at
        clock[0] = 102.1
        assert B._cfg("ollama", "url") == "http://b:2"             # past it: seen

    def test_an_unchanged_file_is_still_not_reparsed_after_the_interval(self, cfg, monkeypatch):
        monkeypatch.setattr(B, "_CONFIG_RECHECK_S", 2.0)
        clock = [0.0]
        monkeypatch.setattr(B.time, "monotonic", lambda: clock[0])
        _write(cfg, '[ollama]\nurl = "http://a:1"\n')
        calls = []
        real = tomllib.loads
        monkeypatch.setattr(tomllib, "loads", lambda text: (calls.append(1), real(text))[1])
        for t in (1.0, 5.0, 9.0, 20.0):
            clock[0] = t
            assert B._config()["ollama"]["url"] == "http://a:1"
        assert len(calls) == 1


POOL = '[[models.pool]]\nname = "m:1"\nroles = ["gen"]\n'


class TestAPinReachesARunningProcess:
    """The incident: a pin written to backends.toml must change what an already-running process sends to Ollama."""

    def test_adding_a_pin_changes_the_requests_options(self, cfg):
        _write(cfg, POOL, mtime_ns=1_000_000_000)
        assert model_pool.options_for("m:1") == {}
        _write(cfg, POOL + "gpu = 1\n", mtime_ns=2_000_000_000)
        assert model_pool.options_for("m:1") == {"main_gpu": 1}

    def test_removing_a_pin_takes_it_back_out(self, cfg):
        _write(cfg, POOL + "gpu = 0\n", mtime_ns=1_000_000_000)
        assert model_pool.options_for("m:1") == {"main_gpu": 0}
        _write(cfg, POOL, mtime_ns=2_000_000_000)
        assert model_pool.options_for("m:1") == {}

    def test_two_readers_of_one_file_agree(self, cfg):
        """Two callers (the server and a CLI tool) read the same file after an edit and send the same thing."""
        _write(cfg, POOL, mtime_ns=1_000_000_000)
        first = model_pool.options_for("m:1")
        _write(cfg, POOL + "gpu = 1\n", mtime_ns=2_000_000_000)
        assert model_pool.options_for("m:1") == {"main_gpu": 1} and first == {}
        B._config.cache_clear()                                 # a freshly started process
        assert model_pool.options_for("m:1") == {"main_gpu": 1}
