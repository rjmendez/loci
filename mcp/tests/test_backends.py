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


# --- Fresh-install end-to-end: no env overrides, no local Ollama/vLLM, config resolved from
# a file (or absent). The in-process equivalent of the env -i simulation, exercising ALL
# backends together so the resolution chain + the bge default flip are guarded as one unit. ---

_ALL_BACKEND_ENV = ("OLLAMA_BASE_URL", "OLLAMA_URL", "VLLM_BASE_URL", "EMBED_MODEL",
                    "VLLM_MODEL", "RERANK_MODEL", "QDRANT_URL", "QDRANT_API_KEY",
                    "LOCI_MEMORY_MD_DIR", "HERMES_MEMORY_DIR")


def _fresh_install(mp, config_path):
    """Clean machine: no backend env overrides, no local service listening, config at path."""
    for k in _ALL_BACKEND_ENV:
        mp.delenv(k, raising=False)
    mp.setattr(B, "_alive", lambda url, timeout=1.0: False)   # no local GPU services
    mp.setattr(B, "_CONFIG_PATH", str(config_path))
    B._reset_cache()


def test_fresh_install_full_config_resolves_all_backends(tmp_path, monkeypatch):
    cfg = tmp_path / "backends.toml"
    cfg.write_text('[ollama]\nurl = "http://cfg-gpu:11434"\n'
                   '[embed]\nmodel = "cfg-embed"\n'
                   '[vllm]\nurl = "http://cfg-gpu:8000"\nmodel = "cfg-vllm"\n'
                   '[rerank]\nmodel = "cfg-rerank"\n'
                   '[qdrant]\nurl = "http://cfg-qdrant:6333"\napi_key = "cfg-key"\n'
                   '[memory]\ndir = "/cfg/mem"\n')
    _fresh_install(monkeypatch, cfg)
    assert B.ollama_url() == "http://cfg-gpu:11434"
    assert B.vllm_url() == "http://cfg-gpu:8000"
    assert B.embed_model() == "cfg-embed"
    assert B.vllm_model() == "cfg-vllm"
    assert B.rerank_model() == "cfg-rerank"
    assert B.qdrant() == ("http://cfg-qdrant:6333", "cfg-key")
    assert B.memory_dir() == "/cfg/mem"


def test_fresh_install_bare_defaults(monkeypatch):
    _fresh_install(monkeypatch, "/nonexistent/no.toml")
    assert B.ollama_url() == "" and B.vllm_url() == ""        # urls empty -> tiers fail-open
    assert B.embed_model() == "nomic-embed-text"
    assert B.vllm_model() == "Qwen2.5-3B-Instruct"
    assert B.rerank_model() == "BAAI/bge-reranker-v2-m3"      # bge is the flipped fresh default
    assert B.qdrant() == ("", "")
    assert B.memory_dir() == ""
