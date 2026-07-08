"""Tests for backends.py — the portable env -> local-probe -> config -> default resolution."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import backends as B  # noqa: E402


def _no_ollama_env(mp):
    mp.delenv("OLLAMA_BASE_URL", raising=False)
    mp.delenv("OLLAMA_URL", raising=False)


def test_env_wins(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://envhost:11434")
    B._reset_cache()
    assert B.ollama_url() == "http://envhost:11434"   # no probe, no config consulted


def test_local_probe_used_when_no_env(monkeypatch):
    _no_ollama_env(monkeypatch)
    monkeypatch.setattr(B, "_alive", lambda url, timeout=1.0: url == B._LOCAL_OLLAMA)
    monkeypatch.setattr(B, "_CONFIG_PATH", "/nonexistent")
    B._reset_cache()
    assert B.ollama_url() == B._LOCAL_OLLAMA        # laptop auto-uses its own GPU


def test_config_fallback_when_no_local(tmp_path, monkeypatch):
    _no_ollama_env(monkeypatch)
    monkeypatch.setattr(B, "_alive", lambda url, timeout=1.0: False)   # no local GPU
    cfg = tmp_path / "backends.toml"
    cfg.write_text('[ollama]\nurl = "http://remote:11434"\n')
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    assert B.ollama_url() == "http://remote:11434"   # headless -> remote (tailscale) infra


def test_empty_when_nothing_configured(monkeypatch):
    _no_ollama_env(monkeypatch)
    monkeypatch.setattr(B, "_alive", lambda url, timeout=1.0: False)
    monkeypatch.setattr(B, "_CONFIG_PATH", "/nonexistent")
    B._reset_cache()
    assert B.ollama_url() == ""                       # fail-open: tiers degrade on empty


def test_models_qdrant_memory_from_config(tmp_path, monkeypatch):
    for k in ("EMBED_MODEL", "VLLM_MODEL", "RERANK_MODEL", "QDRANT_URL",
              "QDRANT_API_KEY", "LOCI_MEMORY_MD_DIR", "HERMES_MEMORY_DIR"):
        monkeypatch.delenv(k, raising=False)
    cfg = tmp_path / "b.toml"
    cfg.write_text('[embed]\nmodel="e"\n[vllm]\nmodel="v"\n[rerank]\nmodel="r"\n'
                   '[qdrant]\nurl="q"\napi_key="k"\n[memory]\ndir="/m"\n')
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    assert B.embed_model() == "e" and B.vllm_model() == "v" and B.rerank_model() == "r"
    assert B.qdrant() == ("q", "k") and B.memory_dir() == "/m"


def test_env_overrides_config_for_models(tmp_path, monkeypatch):
    cfg = tmp_path / "b.toml"
    cfg.write_text('[embed]\nmodel="cfg"\n')
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    monkeypatch.setenv("EMBED_MODEL", "envmodel")
    assert B.embed_model() == "envmodel"


def test_broken_config_is_fail_open(tmp_path, monkeypatch):
    for k in ("EMBED_MODEL", "OLLAMA_BASE_URL", "OLLAMA_URL"):
        monkeypatch.delenv(k, raising=False)
    cfg = tmp_path / "bad.toml"
    cfg.write_text("this is [not valid toml")
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    monkeypatch.setattr(B, "_alive", lambda url, timeout=1.0: False)
    B._reset_cache()
    assert B.embed_model() == "nomic-embed-text"      # falls to default, never raises
    assert B.ollama_url() == ""
