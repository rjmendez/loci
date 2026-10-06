"""gpu_warm.py's default gen model must track backends.ollama_gen_model(),
not a hardcoded "qwen2.5:3b" that can silently diverge from the model
mcp/llm_local.py actually generates with (mcp/memcheck/llm.py._llm_model()
already resolves through backends the same way).
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys
from unittest import mock


REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "mcp"))


def _fresh_gpu_warm():
    sys.modules.pop("gpu_warm", None)
    sys.modules.pop("backends", None)
    return importlib.import_module("gpu_warm")


def test_gen_model_follows_backends(tmp_path):
    """No WARM_GEN_MODEL override -> resolve via backends.ollama_gen_model() (here through its per-process
    env override; in service the model pool decides)."""
    cfg = tmp_path / "backends.toml"
    cfg.write_text("[ollama]\n")
    env = dict(os.environ)
    env.pop("WARM_GEN_MODEL", None)
    env["LOCI_CONFIG"] = str(cfg)
    env["LOCI_OLLAMA_GEN_MODEL"] = "heretic-llama31-8b-instruct:latest"
    with mock.patch.dict(os.environ, env, clear=True):
        mod = _fresh_gpu_warm()
        assert mod._GEN_MODEL == "heretic-llama31-8b-instruct:latest"


def test_a_gen_model_named_in_config_is_not_warmed(tmp_path):
    """Models are never named in config: a [ollama].gen_model key pins nothing, so gpu_warm cannot warm it."""
    cfg = tmp_path / "backends.toml"
    cfg.write_text('[ollama]\ngen_model = "cfg-only-tag-that-is-not-installed:1b"\n')
    env = dict(os.environ)
    env.pop("WARM_GEN_MODEL", None)
    env.pop("LOCI_OLLAMA_GEN_MODEL", None)
    env["LOCI_CONFIG"] = str(cfg)
    with mock.patch.dict(os.environ, env, clear=True):
        mod = _fresh_gpu_warm()
        assert mod._GEN_MODEL != "cfg-only-tag-that-is-not-installed:1b"


def test_explicit_warm_gen_model_env_still_wins(tmp_path):
    cfg = tmp_path / "backends.toml"
    cfg.write_text('[ollama]\ngen_model = "heretic-llama31-8b-instruct:latest"\n')
    env = dict(os.environ)
    env["LOCI_CONFIG"] = str(cfg)
    env["WARM_GEN_MODEL"] = "explicit-override:latest"
    with mock.patch.dict(os.environ, env, clear=True):
        mod = _fresh_gpu_warm()
        assert mod._GEN_MODEL == "explicit-override:latest"
