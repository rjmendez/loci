"""deploy_abliterated_model.py — opt-in heavier Ollama model deploy helper.

These stay pure-unit: no live Hugging Face, no live Ollama, no giant files.
"""
from __future__ import annotations

import errno
import hashlib
import importlib.util
import pathlib
import subprocess
import sys

import pytest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "deploy_abliterated_model", SCRIPTS / "deploy_abliterated_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _test_plan(mod, tmp_path: pathlib.Path, *, payload: bytes = b"good-model", expected_sha256: str | None = None):
    digest = expected_sha256
    if digest is None:
        digest = hashlib.sha256(payload).hexdigest()
    spec = mod.ModelSpec(
        choice="toy-model",
        repo="acme/toy-model",
        template="Modelfile.toy-model",
        filename_prefix="toy-model",
        tag_prefix="loci-toy-model",
        allowed_quants=("q4_k_m",),
        default_quant="q4_k_m",
        expected_bytes={"q4_k_m": len(payload)},
        expected_sha256=({} if expected_sha256 is None else {"q4_k_m": digest}),
    )
    root = tmp_path / "models" / spec.choice / "q4_k_m"
    return mod.Plan(
        spec=spec,
        quant="q4_k_m",
        tag="loci-toy-model:q4km",
        template_path=tmp_path / spec.template,
        gguf_path=root / spec.filename("q4_k_m"),
        local_modelfile=root / f"{spec.template}.local",
        expected_bytes=len(payload),
        expected_sha256=digest if expected_sha256 is not None else None,
    )


def test_build_plan_uses_researched_defaults():
    mod = _load()
    plan = mod.build_plan("14b-coder", quant=None, models_dir=pathlib.Path("/models"), tag=None)
    assert plan.quant == "q4_k_m"
    assert plan.tag == "loci-qwen25-coder-14b-abliterated:q4km"
    assert plan.expected_bytes == 8988111200
    assert plan.expected_sha256 == "e89a7ae4e2b456bf33c75cff35664751df20ff273e551d7cf7640aa9e84d3b79"
    assert plan.gguf_path.name == "Qwen2.5-Coder-14B-Instruct-abliterated-Q4_K_M.gguf"


def test_build_plan_includes_qwen38_defaults():
    mod = _load()
    plan = mod.build_plan("27b-qwen38", quant=None, models_dir=pathlib.Path("/models"), tag=None)
    assert plan.quant == "q5_k"
    assert plan.tag == "loci-qwen38-27b-abliterated:q5k"
    assert plan.expected_bytes == 19535701280
    assert plan.expected_sha256 == "917453854fc640903f89bda0b29eb7ea661cb13b9c5c9efd20a68f3ff2e9277f"
    assert plan.gguf_path.name == "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"


def test_build_plan_includes_gemma4_defaults():
    mod = _load()
    plan = mod.build_plan("26b-gemma4", quant=None, models_dir=pathlib.Path("/models"), tag=None)
    assert plan.quant == "q4_k_m"
    assert plan.tag == "loci-gemma4-26b-a4b-abliterated:q4km"
    assert plan.expected_bytes == 16868236224
    assert plan.expected_sha256 == "e049f67e6d4f22700f39b8018f7612d455151891dd4e5cbac06f13ce3b6e83f5"
    assert plan.gguf_path.name == "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"


def test_render_modelfile_rewrites_from_line_only():
    mod = _load()
    rendered = mod._render_modelfile("# comment\nFROM ./old.gguf\nPARAMETER temperature 0.2\n",
                                     pathlib.Path("/weights/model.gguf"))
    assert rendered.startswith("# comment\nFROM /weights/model.gguf\n")
    assert "PARAMETER temperature 0.2" in rendered


def test_main_dry_run_parses_and_prints_plan(capsys):
    mod = _load()
    rc = mod.main(["24b-mistral", "--dry-run", "--models-dir", "/models"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "loci-mistral-small-24b-abliterated:q5km" in out
    assert "LOCI_OLLAMA_GEN_MODEL" in out


def test_main_dry_run_supports_new_registry_entries(capsys):
    mod = _load()
    rc = mod.main(["27b-qwen38", "--dry-run", "--models-dir", "/models"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "loci-qwen38-27b-abliterated:q5k" in out

    rc = mod.main(["26b-gemma4", "--dry-run", "--models-dir", "/models"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "loci-gemma4-26b-a4b-abliterated:q4km" in out


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


def test_main_aborts_before_create_when_existing_checksum_mismatches(tmp_path, monkeypatch, capsys):
    mod = _load()
    plan = _test_plan(mod, tmp_path, payload=b"abcd", expected_sha256="0" * 64)
    plan.gguf_path.parent.mkdir(parents=True, exist_ok=True)
    plan.gguf_path.write_bytes(b"abcd")
    monkeypatch.setattr(mod, "build_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(mod, "_require_tool", lambda names, label: names[0])
    monkeypatch.setattr(mod, "_ensure_ollama_ready", lambda url: None)
    monkeypatch.setattr(mod, "_write_modelfile",
                        lambda *args, **kwargs: pytest.fail("checksum failure must abort before Modelfile write"))
    monkeypatch.setattr(mod, "_create_model",
                        lambda *args, **kwargs: pytest.fail("checksum failure must abort before ollama create"))
    rc = mod.main(["14b-coder", "--models-dir", str(tmp_path / "models")])
    assert rc == 2
    assert "SHA256 mismatch" in capsys.readouterr().err


def test_verify_checksum_requires_manifest_entry(tmp_path):
    mod = _load()
    plan = _test_plan(mod, tmp_path, payload=b"abcd")
    plan.gguf_path.parent.mkdir(parents=True, exist_ok=True)
    plan.gguf_path.write_bytes(b"abcd")

    with pytest.raises(RuntimeError, match="missing SHA256 manifest entry"):
        mod._verify_checksum(plan)


def test_download_cleans_partial_file_on_failure(tmp_path, monkeypatch):
    mod = _load()
    payload = b"abcd"
    plan = _test_plan(mod, tmp_path, payload=payload, expected_sha256=hashlib.sha256(payload).hexdigest())
    stage_dir = mod._stage_dir(plan.gguf_path)
    part_path = mod._part_path(plan.gguf_path)

    def fake_run(cmd, **kwargs):
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / plan.gguf_path.name).write_bytes(payload[:2])
        return subprocess.CompletedProcess(cmd, 1, "", "stub download failed")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="download failed"):
        mod._download("hf", plan)

    assert not plan.gguf_path.exists()
    assert not part_path.exists()
    assert not stage_dir.exists()


def test_download_cleans_partial_file_on_enospc(tmp_path, monkeypatch):
    mod = _load()
    payload = b"abcd"
    digest = hashlib.sha256(payload).hexdigest()
    plan = _test_plan(mod, tmp_path, payload=payload, expected_sha256=digest)
    stage_dir = mod._stage_dir(plan.gguf_path)
    part_path = mod._part_path(plan.gguf_path)
    real_replace = mod.os.replace

    def fake_run(cmd, **kwargs):
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / plan.gguf_path.name).write_bytes(payload)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def fake_replace(src, dst):
        if pathlib.Path(dst) == plan.gguf_path:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_replace(src, dst)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod.os, "replace", fake_replace)

    with pytest.raises(RuntimeError, match="ran out of disk space"):
        mod._download("hf", plan)

    assert not plan.gguf_path.exists()
    assert not part_path.exists()
    assert not stage_dir.exists()


def test_smoke_test_rejects_wrong_content(tmp_path, monkeypatch):
    mod = _load()
    plan = _test_plan(mod, tmp_path, expected_sha256="f" * 64)
    monkeypatch.setattr(mod, "_json_request", lambda *args, **kwargs: {"response": "totally unrelated"})

    with pytest.raises(RuntimeError, match="expected marker"):
        mod._smoke_test(plan, base_url="http://ollama.test", keep_alive="30m")
