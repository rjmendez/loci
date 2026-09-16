"""deploy_abliterated_model.py — opt-in heavier Ollama model deploy helper.

These stay pure-unit: no live Hugging Face, no live Ollama, no giant files.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "deploy_abliterated_model", SCRIPTS / "deploy_abliterated_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_plan_uses_researched_defaults():
    mod = _load()
    plan = mod.build_plan("14b-coder", quant=None, models_dir=pathlib.Path("/models"), tag=None)
    assert plan.quant == "q4_k_m"
    assert plan.tag == "loci-qwen25-coder-14b-abliterated:q4km"
    assert plan.expected_bytes == 8988111200
    assert plan.gguf_path.name == "Qwen2.5-Coder-14B-Instruct-abliterated-Q4_K_M.gguf"


def test_render_modelfile_rewrites_from_line_only():
    mod = _load()
    rendered = mod._render_modelfile("FROM ./old.gguf\nPARAMETER temperature 0.2\n",
                                     pathlib.Path("/weights/model.gguf"))
    assert rendered.startswith("FROM /weights/model.gguf\n")
    assert "PARAMETER temperature 0.2" in rendered


def test_main_dry_run_parses_and_prints_plan(capsys):
    mod = _load()
    rc = mod.main(["24b-mistral", "--dry-run", "--models-dir", "/models"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "loci-mistral-small-24b-abliterated:q5km" in out
    assert "LOCI_OLLAMA_GEN_MODEL" in out


def test_main_refuses_cleanly_on_low_disk(monkeypatch, capsys):
    mod = _load()
    monkeypatch.setattr(mod, "_require_tool", lambda names, label: names[0])
    monkeypatch.setattr(mod, "_ensure_ollama_ready", lambda url: None)
    monkeypatch.setattr(mod, "_existing_file_state", lambda path, expected: "missing")
    monkeypatch.setattr(mod.shutil, "disk_usage",
                        lambda path: type("DU", (), {"free": 1, "used": 0, "total": 1})())
    rc = mod.main(["14b-coder", "--models-dir", "/models"])
    assert rc == 2
    assert "insufficient disk" in capsys.readouterr().err
