"""Tests for scripts/loci_setup_verify.py apply/verify behaviors."""
from __future__ import annotations

import importlib.util
import io
import pathlib
from contextlib import redirect_stdout


REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "loci_setup_verify.py"


def _load():
    spec = importlib.util.spec_from_file_location("loci_setup_verify", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(mod, argv):
    out = io.StringIO()
    with redirect_stdout(out):
        code = mod.main(argv)
    return code, out.getvalue()


def test_key_parsing_never_leaks_values(tmp_path):
    mod = _load()
    cfg = tmp_path / "backends.toml"
    or_file = tmp_path / "openrouter.key"
    ab_file = tmp_path / "abliteration.key"
    or_file.write_text('export OPENROUTER_API_KEY="or-secret-123"\n', encoding="utf-8")
    ab_file.write_text("ABLITERATION_API_KEY=ab-secret-456\n", encoding="utf-8")

    rc, output = _run(
        mod,
        [
            "--apply",
            "--config",
            str(cfg),
            "--cloud-tier",
            "on",
            "--openrouter-url",
            "https://openrouter.ai/api/v1",
            "--abliteration-url",
            "https://ablit.invalid/v1",
            "--openrouter-key-file",
            str(or_file),
            "--abliteration-key-file",
            str(ab_file),
        ],
    )
    assert rc == 0, output
    assert "openrouter=present" in output
    assert "abliteration=present" in output
    assert "or-secret-123" not in output
    assert "ab-secret-456" not in output


def test_apply_mode_writes_expected_sections_and_fields(tmp_path):
    mod = _load()
    cfg = tmp_path / "backends.toml"
    or_file = tmp_path / "openrouter.key"
    ab_file = tmp_path / "abliteration.key"
    or_file.write_text("or-key\n", encoding="utf-8")
    ab_file.write_text("ab-key\n", encoding="utf-8")

    rc, output = _run(
        mod,
        [
            "--apply",
            "--config",
            str(cfg),
            "--cloud-tier",
            "on",
            "--openrouter-url",
            "https://openrouter.ai/api/v1",
            "--openrouter-model",
            "openai/gpt-5-mini",
            "--openrouter-role-model",
            "triage=openai/gpt-5-nano",
            "--abliteration-url",
            "https://ablit.invalid/v1",
            "--abliteration-model",
            "zai-org/GLM-4.5",
            "--abliteration-role-model",
            "redteam=moonshotai/kimi-k2",
            "--tmux-offload",
            "on",
            "--tmux-role-session",
            "triage=tmux-triage",
            "--tmux-expensive-roles",
            "reasoning,redteam",
            "--tmux-require-mapped-session",
            "on",
            "--openrouter-key-file",
            str(or_file),
            "--abliteration-key-file",
            str(ab_file),
        ],
    )
    assert rc == 1, output  # tmux strict mode verifies and fails without tmux session

    data = mod._read_toml(cfg)
    assert data["cloud"]["enabled"] is True
    assert data["openrouter"]["url"] == "https://openrouter.ai/api/v1"
    assert data["openrouter"]["model"] == "openai/gpt-5-mini"
    assert data["openrouter"]["triage"]["model"] == "openai/gpt-5-nano"
    assert data["openrouter"]["key"] == "or-key"
    assert data["abliteration"]["url"] == "https://ablit.invalid/v1"
    assert data["abliteration"]["model"] == "zai-org/GLM-4.5"
    assert data["abliteration"]["redteam"]["model"] == "moonshotai/kimi-k2"
    assert data["abliteration"]["key"] == "ab-key"
    assert data["tmux_offload"]["enabled"] is True
    assert data["tmux_offload"]["require_mapped_session"] is True
    assert data["tmux_offload"]["role_sessions"]["triage"] == "tmux-triage"
    assert data["tmux_offload"]["expensive_roles"] == ["reasoning", "redteam"]


def test_verify_mode_fails_on_missing_prerequisites(tmp_path):
    mod = _load()
    cfg = tmp_path / "backends.toml"
    cfg.write_text(
        "[cloud]\n"
        "enabled=true\n\n"
        "[openrouter]\n"
        'key="present-but-no-url"\n',
        encoding="utf-8",
    )

    rc, output = _run(mod, ["--config", str(cfg)])
    assert rc == 1
    assert "openrouter key set but [openrouter].url missing" in output


def test_tmux_verification_uses_stubbed_session_check(tmp_path, monkeypatch):
    mod = _load()
    cfg = tmp_path / "backends.toml"
    cfg.write_text(
        "[tmux_offload]\n"
        "enabled=true\n"
        "require_mapped_session=true\n"
        'role_sessions={ triage="triage-sess", redteam="red-sess" }\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mod,
        "_tmux_session_exists",
        lambda name: name == "triage-sess",
    )

    rc, output = _run(mod, ["--config", str(cfg)])
    assert rc == 1
    assert "missing tmux sessions: redteam=red-sess" in output


def test_provider_auth_check_uses_bearer_key(monkeypatch):
    mod = _load()

    captured = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

    def fake_urlopen(req, timeout=0):
        captured["auth"] = req.headers.get("Authorization")
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(mod.request, "urlopen", fake_urlopen)
    ok, msg = mod._provider_auth_check(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "sk-test-123",
    )
    assert ok is True
    assert "status 200" in msg
    assert captured["auth"] == "Bearer sk-test-123"
    assert captured["timeout"] == 12
