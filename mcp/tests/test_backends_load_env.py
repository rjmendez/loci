"""A scheduled run inherits none of the MCP launcher's environment.

On this host the Ollama and Qdrant endpoints existed only inside that launcher's
process, so cron resolved localhost:11434, found nothing listening, and reported
every embedding step "skipped" — which reads as a schedule decision rather than a
missing endpoint. load_env() puts the config file's values into os.environ, where
child processes can see them too.
"""
import os
import sys
import pathlib
import importlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import backends  # noqa: E402


VARS = ("OLLAMA_BASE_URL", "QDRANT_URL", "QDRANT_API_KEY", "EMBED_MODEL",
        "LOCI_QDRANT_RETENTION_DAYS", "OLLAMA_URL")


def _clean(monkeypatch):
    for v in VARS:
        monkeypatch.delenv(v, raising=False)
    backends._reset_cache()


def _config(monkeypatch, tmp_path, body):
    cfg = tmp_path / "backends.toml"
    cfg.write_text(body)
    monkeypatch.setenv("LOCI_CONFIG", str(cfg))
    monkeypatch.setattr(backends, "_CONFIG_PATH", str(cfg))
    monkeypatch.setattr(backends, "_alive", lambda url, timeout=1.0: False)
    backends._reset_cache()


def test_config_file_reaches_the_environment(monkeypatch, tmp_path):
    _clean(monkeypatch)
    _config(monkeypatch, tmp_path, '''
[ollama]
url = "http://gpu-host:11434"
[embed]
model = "some-embed-model"
[qdrant]
url = "http://qdrant-host:6333"
api_key = "sekrit"
retention_days = 0
''')
    resolved = backends.load_env(tmp_path)

    assert os.environ["OLLAMA_BASE_URL"] == "http://gpu-host:11434"
    assert os.environ["QDRANT_URL"] == "http://qdrant-host:6333"
    assert os.environ["QDRANT_API_KEY"] == "sekrit"
    assert os.environ["EMBED_MODEL"] == "some-embed-model"
    assert os.environ["LOCI_QDRANT_RETENTION_DAYS"] == "0"
    assert set(resolved) >= {"OLLAMA_BASE_URL", "QDRANT_URL", "EMBED_MODEL"}


def test_an_existing_environment_variable_always_wins(monkeypatch, tmp_path):
    """A power-user override must survive; the config file is the fallback."""
    _clean(monkeypatch)
    _config(monkeypatch, tmp_path, '[ollama]\nurl = "http://from-config:11434"\n')
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-env:11434")

    resolved = backends.load_env(tmp_path)

    assert os.environ["OLLAMA_BASE_URL"] == "http://from-env:11434"
    assert "OLLAMA_BASE_URL" not in resolved, "must not report what it did not set"


def test_a_missing_config_file_is_not_an_error(monkeypatch, tmp_path):
    """Fail-open: an unconfigured machine still runs, on its own local backends."""
    _clean(monkeypatch)
    monkeypatch.setattr(backends, "_CONFIG_PATH", str(tmp_path / "nope.toml"))
    monkeypatch.setattr(backends, "_alive", lambda url, timeout=1.0: False)
    backends._reset_cache()

    resolved = backends.load_env(tmp_path)
    assert "OLLAMA_BASE_URL" not in resolved
    assert isinstance(resolved, dict)


def test_the_loop_resolves_at_run_time_not_import_time(monkeypatch, tmp_path):
    """An import-time default is fixed before the config file is ever read, and
    mutating os.environ on import leaks into every other module in the process."""
    _clean(monkeypatch)
    repo = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("loop_probe", repo / "mlops" / "loop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert os.environ.get("OLLAMA_BASE_URL") is None, (
        "importing mlops.loop must not write to the environment"
    )
    src = (repo / "mlops" / "loop.py").read_text()
    assert 'ap.add_argument("--ollama", default=None' in src, (
        "--ollama must default to None so the config file gets a say"
    )
