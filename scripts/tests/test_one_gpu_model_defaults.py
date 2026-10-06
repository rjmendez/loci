"""Script model defaults come from env/backends.toml, and never from an oversized literal.

qwen3.8:latest (17.7 GB), gemma4:26b (18.6 GB) and the 27B Qwen3.8 heretic build
(17.2 GB) do not fit one 11-12 GB GPU: Ollama splits them across cards and the loads
time out, blocking its scheduler. These scripts used to carry those tags as literal
defaults that bypassed the operator's config.
"""
from __future__ import annotations

import importlib
import pathlib
import sys

import pytest


REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "mcp"))

_MODULES = ("swarm_escalate", "local_deep_think", "model_catalog")
_MODEL_ENV = (
    "LOCI_SWARM_ESCALATE_MODEL", "LOCI_SWARM_SYNTHESIZE_MODEL",
    "LOCI_OLLAMA_GEN_MODEL", "LOCI_OLLAMA_VERIFY_MODEL", "LOCI_OLLAMA_REDTEAM_MODEL",
    "LOCI_LOCAL_DEEP_THINK_VERIFY_MODEL", "LOCI_LOCAL_DEEP_THINK_SYNTHESIZE_MODEL",
    "LOCI_LOCAL_DEEP_THINK_SELF_REFLECT_MODEL", "LOCI_LOCAL_DEEP_THINK_IDEATE_MODELS",
)
_OVERSIZED = ("qwen3.8", "gemma4:26b", "27b")


@pytest.fixture
def reload_scripts(monkeypatch, tmp_path):
    """Reload the scripts against a given backends.toml (None = no config at all)."""

    def _reload(config_text, env=None):
        # Look backends up at call time: other tests pop it from sys.modules, and the
        # scripts import whichever module object is current.
        backends = importlib.import_module("backends")
        for name in _MODEL_ENV:
            monkeypatch.delenv(name, raising=False)
        for name, value in (env or {}).items():
            monkeypatch.setenv(name, value)
        cfg = tmp_path / "backends.toml"
        if config_text is not None:
            cfg.write_text(config_text)
        monkeypatch.setattr(backends, "_CONFIG_PATH", str(cfg))
        # No local inventory: the defaults must not depend on what this box has pulled.
        monkeypatch.setattr(backends, "_ollama_list", lambda: {})
        monkeypatch.setattr(backends, "_ollama_local_tags", lambda: set())
        backends._reset_cache()
        return [importlib.reload(importlib.import_module(name)) for name in _MODULES]

    yield _reload
    monkeypatch.undo()
    importlib.import_module("backends")._reset_cache()
    for name in _MODULES:
        if name in sys.modules:
            importlib.reload(sys.modules[name])


def _assert_one_gpu(model: str) -> None:
    assert model and not any(marker in model.lower() for marker in _OVERSIZED), model


def test_script_defaults_follow_backends(reload_scripts):
    """The script defaults are whatever backends resolves (here via the per-process env override; in service
    the model pool decides)."""
    swarm, deep, catalog = reload_scripts(
        None, env={"LOCI_OLLAMA_GEN_MODEL": "gemma4-e4b-hermes:64k", "LOCI_OLLAMA_VERIFY_MODEL": "gemma4-e4b-hermes:64k"})
    assert swarm._DEFAULT_ESCALATE_MODEL == "gemma4-e4b-hermes:64k"
    assert swarm._DEFAULT_SYNTHESIZE_MODEL == "gemma4-e4b-hermes:64k"
    assert deep._DEFAULT_VERIFY_MODEL == "gemma4-e4b-hermes:64k"
    assert deep._DEFAULT_SYNTH_MODEL == "gemma4-e4b-hermes:64k"
    assert deep._DEFAULT_REFLECT_MODEL == "gemma4-e4b-hermes:64k"
    assert deep._DEFAULT_IDEATE_MODELS == "llama3.1-agent:latest,gemma4-e4b-hermes:64k"
    assert catalog.SWARM_ESCALATION_MODEL == "gemma4-e4b-hermes:64k"
    assert catalog.SWARM_SYNTHESIS_MODEL == "gemma4-e4b-hermes:64k"

    # The CLI paths resolve to the same values when no flag is passed.
    args = swarm.parse_args(["topic"])
    config = swarm._resolve_config(args)
    assert config.escalate_model == "gemma4-e4b-hermes:64k"
    assert config.synthesize_model == "gemma4-e4b-hermes:64k"
    chain = deep._resolve_models(deep.parse_args(["topic"]))
    assert chain.verify_model == "gemma4-e4b-hermes:64k"
    assert chain.synthesize_model == "gemma4-e4b-hermes:64k"
    assert chain.self_reflect_model == "gemma4-e4b-hermes:64k"


def test_swarm_specific_env_wins_over_the_verify_tier(reload_scripts):
    swarm, _, catalog = reload_scripts(None, env={
        "LOCI_OLLAMA_VERIFY_MODEL": "v:4b",
        "LOCI_SWARM_ESCALATE_MODEL": "esc:4b", "LOCI_SWARM_SYNTHESIZE_MODEL": "syn:4b"})
    assert swarm._DEFAULT_ESCALATE_MODEL == catalog.SWARM_ESCALATION_MODEL == "esc:4b"
    assert swarm._DEFAULT_SYNTHESIZE_MODEL == catalog.SWARM_SYNTHESIS_MODEL == "syn:4b"


def test_swarm_models_named_in_config_are_ignored(reload_scripts):
    """Config pins nothing: the swarm tiers follow the verify tier, which with no pool and nothing installed is
    the last-resort default."""
    swarm, _, catalog = reload_scripts(
        '[ollama]\nverify_model = "v:4b"\nswarm_escalate_model = "esc:4b"\nswarm_synthesize_model = "syn:4b"\n')
    backends = importlib.import_module("backends")
    assert swarm._DEFAULT_ESCALATE_MODEL == catalog.SWARM_ESCALATION_MODEL == backends.ONE_GPU_FALLBACK_MODEL
    assert swarm._DEFAULT_SYNTHESIZE_MODEL == catalog.SWARM_SYNTHESIS_MODEL == backends.ONE_GPU_FALLBACK_MODEL


def test_swarm_env_still_wins_over_config(reload_scripts, monkeypatch):
    reload_scripts('[ollama]\nswarm_escalate_model = "cfg:4b"\n')
    monkeypatch.setenv("LOCI_SWARM_ESCALATE_MODEL", "env-esc:4b")
    monkeypatch.setenv("LOCI_SWARM_SYNTHESIZE_MODEL", "env-syn:4b")
    swarm = importlib.reload(importlib.import_module("swarm_escalate"))
    assert swarm._DEFAULT_ESCALATE_MODEL == "env-esc:4b"
    assert swarm._DEFAULT_SYNTHESIZE_MODEL == "env-syn:4b"


def test_no_config_no_env_never_resolves_an_oversized_model(reload_scripts):
    swarm, deep, catalog = reload_scripts(None)
    defaults = [
        swarm._DEFAULT_CHEAP_MODEL,
        swarm._DEFAULT_ESCALATE_MODEL,
        swarm._DEFAULT_SYNTHESIZE_MODEL,
        swarm._TIER_ESCALATE_MODEL,
        swarm._TIER_SYNTHESIZE_MODEL,
        deep._DEFAULT_VERIFY_MODEL,
        deep._DEFAULT_SYNTH_MODEL,
        deep._DEFAULT_REFLECT_MODEL,
        deep._TIER_VERIFY_MODEL,
        deep._TIER_SYNTH_MODEL,
        deep._TIER_REFLECT_MODEL,
        catalog.SWARM_ESCALATION_MODEL,
        catalog.SWARM_SYNTHESIS_MODEL,
        *catalog.SAFETY_SPECIALIST_MODELS,
        *deep._DEFAULT_IDEATE_MODELS.split(","),
    ]
    for model in defaults:
        _assert_one_gpu(model)
    chain = deep._resolve_models(deep.parse_args(["topic", "--red-team"]))
    for model in (chain.verify_model, chain.synthesize_model, chain.self_reflect_model,
                  chain.redteam_model, *chain.ideate_models):
        _assert_one_gpu(model)
    tiered = swarm._resolve_config(swarm.parse_args(["topic", "--seeds", "2"]))
    _assert_one_gpu(tiered.escalate_model)
    _assert_one_gpu(tiered.synthesize_model)


def test_scripts_fail_open_when_backends_lookup_raises(reload_scripts, monkeypatch):
    reload_scripts(None)
    backends = importlib.import_module("backends")
    monkeypatch.setattr(backends, "swarm_escalate_model", _boom)
    monkeypatch.setattr(backends, "ollama_verify_model", _boom)
    swarm = importlib.reload(importlib.import_module("swarm_escalate"))
    deep = importlib.reload(importlib.import_module("local_deep_think"))
    assert swarm._DEFAULT_ESCALATE_MODEL == "qwen2.5:3b"
    assert deep._DEFAULT_VERIFY_MODEL == "qwen2.5:3b"


def _boom():
    raise RuntimeError("backends unavailable")
