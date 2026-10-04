"""Tests for backends.py — the portable env -> config -> local-probe -> default resolution."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import backends as B  # noqa: E402
import pytest  # noqa: E402

_REAL_OLLAMA_LIST = B._ollama_list


@pytest.fixture(autouse=True)
def _no_real_ollama_list(monkeypatch):
    # Size lookups must not shell out to a real `ollama list` on a dev box.
    monkeypatch.setattr(B, "_ollama_list", lambda: {})


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
    for k in ("EMBED_MODEL", "RERANK_MODEL", "QDRANT_URL", "QDRANT_API_KEY",
              "LOCI_MEMORY_MD_DIR", "LOCI_MEMORY_DIR"):
        monkeypatch.delenv(k, raising=False)
    cfg = tmp_path / "b.toml"
    cfg.write_text('[embed]\nmodel="e"\n[rerank]\nmodel="r"\n'
                   '[qdrant]\nurl="q"\napi_key="k"\n[memory]\ndir="/m"\n')
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    assert B.embed_model() == "e" and B.rerank_model() == "r"
    assert B.qdrant() == ("q", "k") and B.memory_dir() == "/m"


def test_openrouter_and_abliteration_resolve_from_config(tmp_path, monkeypatch):
    for k in ("OPENROUTER_BASE_URL", "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
              "ABLITERATION_BASE_URL", "ABLITERATION_API_KEY", "ABLITERATION_MODEL"):
        monkeypatch.delenv(k, raising=False)
    cfg = tmp_path / "b.toml"
    cfg.write_text(
        '[openrouter]\nurl="https://or"\nkey="ork"\nmodel="or-default"\n'
        '[openrouter.redteam]\nmodel="or-red"\n'
        '[abliteration]\nurl="https://ab"\nkey="abk"\nmodel="ab-default"\n'
        '[abliteration.redteam]\nmodel="ab-red"\n'
    )
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    assert B.openrouter() == ("https://or", "ork")
    assert B.openrouter_model() == "or-default"
    assert B.openrouter_model("redteam") == "or-red"
    assert B.abliteration() == ("https://ab", "abk")
    assert B.abliteration_model() == "ab-default"
    assert B.abliteration_model("redteam") == "ab-red"


def test_cloud_tier_enabled_accepts_bool_and_string(monkeypatch, tmp_path):
    cfg = tmp_path / "b.toml"
    cfg.write_text("[cloud]\nenabled=true\n")
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    assert B.cloud_tier_enabled() is True
    monkeypatch.setenv("LOCI_CLOUD_TIER_ENABLED", "0")
    assert B.cloud_tier_enabled() is False


def test_cloud_guardrail_settings_resolve_from_config_then_env(monkeypatch, tmp_path):
    cfg = tmp_path / "b.toml"
    cfg.write_text(
        "[cloud]\n"
        "max_tokens_per_call=320\n"
        "daily_call_budget=11\n"
        "daily_token_budget=8000\n"
        "deny_providers=[\"abliteration\"]\n"
        "deny_roles=[\"redteam\"]\n"
        "allowed_roles=[\"triage\",\"coding\"]\n"
        "budget_state_path=\"C:\\\\state\\\\cloud-budget.json\"\n"
    )
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    assert B.cloud_max_tokens_per_call() == 320
    assert B.cloud_daily_call_budget() == 11
    assert B.cloud_daily_token_budget() == 8000
    assert B.cloud_deny_providers() == {"abliteration"}
    assert B.cloud_deny_roles() == {"redteam"}
    assert B.cloud_allowed_roles() == {"triage", "coding"}
    assert B.cloud_budget_state_path() == "C:\\state\\cloud-budget.json"

    monkeypatch.setenv("LOCI_CLOUD_TIER_MAX_TOKENS_PER_CALL", "64")
    monkeypatch.setenv("LOCI_CLOUD_TIER_DAILY_CALL_BUDGET", "2")
    monkeypatch.setenv("LOCI_CLOUD_TIER_DAILY_TOKEN_BUDGET", "700")
    monkeypatch.setenv("LOCI_CLOUD_TIER_DENY_PROVIDERS", "openrouter")
    monkeypatch.setenv("LOCI_CLOUD_TIER_DENY_ROLES", "synthesis")
    monkeypatch.setenv("LOCI_CLOUD_TIER_ALLOWED_ROLES", "reasoning,triage")
    monkeypatch.setenv("LOCI_CLOUD_TIER_BUDGET_STATE_PATH", "C:\\state\\cloud.json")
    assert B.cloud_max_tokens_per_call() == 64
    assert B.cloud_daily_call_budget() == 2
    assert B.cloud_daily_token_budget() == 700
    assert B.cloud_deny_providers() == {"openrouter"}
    assert B.cloud_deny_roles() == {"synthesis"}
    assert B.cloud_allowed_roles() == {"reasoning", "triage"}
    assert B.cloud_budget_state_path() == "C:\\state\\cloud.json"


def test_tmux_offload_defaults_preserve_current_behavior(monkeypatch):
    for key in (
        "LOCI_TMUX_OFFLOAD_ENABLED",
        "LOCI_TMUX_OFFLOAD_ROLE_SESSIONS",
        "LOCI_TMUX_OFFLOAD_EXPENSIVE_ROLES",
        "LOCI_TMUX_OFFLOAD_REQUIRE_MAPPED_SESSION",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(B, "_CONFIG_PATH", "/nonexistent")
    B._reset_cache()
    assert B.tmux_offload_enabled() is False
    assert B.tmux_offload_role_sessions() == {}
    assert B.tmux_offload_expensive_roles() == set()
    assert B.tmux_offload_require_mapped_session() is False


def test_tmux_offload_policy_resolves_from_config_then_env(monkeypatch, tmp_path):
    cfg = tmp_path / "b.toml"
    cfg.write_text(
        "[tmux_offload]\n"
        "enabled=true\n"
        "role_sessions={triage=\"lane-fast\",reasoning=\"lane-deep\"}\n"
        "expensive_roles=[\"reasoning\",\"synthesis\"]\n"
        "require_mapped_session=true\n"
    )
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    assert B.tmux_offload_enabled() is True
    assert B.tmux_offload_role_sessions() == {"triage": "lane-fast", "reasoning": "lane-deep"}
    assert B.tmux_offload_expensive_roles() == {"reasoning", "synthesis"}
    assert B.tmux_offload_require_mapped_session() is True

    monkeypatch.setenv("LOCI_TMUX_OFFLOAD_ENABLED", "0")
    monkeypatch.setenv("LOCI_TMUX_OFFLOAD_ROLE_SESSIONS", "triage=lane-a,redteam=lane-r")
    monkeypatch.setenv("LOCI_TMUX_OFFLOAD_EXPENSIVE_ROLES", "redteam,reasoning")
    monkeypatch.setenv("LOCI_TMUX_OFFLOAD_REQUIRE_MAPPED_SESSION", "false")
    assert B.tmux_offload_enabled() is False
    assert B.tmux_offload_role_sessions() == {"triage": "lane-a", "redteam": "lane-r"}
    assert B.tmux_offload_expensive_roles() == {"redteam", "reasoning"}
    assert B.tmux_offload_require_mapped_session() is False


def test_memory_dir_env_beats_config_even_for_legacy_name(tmp_path, monkeypatch):
    for k in ("LOCI_MEMORY_MD_DIR", "LOCI_MEMORY_DIR", "HERMES_MEMORY_DIR"):
        monkeypatch.delenv(k, raising=False)
    cfg = tmp_path / "b.toml"
    cfg.write_text('[memory]\ndir="/cfg/mem"\n')
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    monkeypatch.setenv("HERMES_MEMORY_DIR", "/legacy/mem")
    assert B.memory_dir() == "/legacy/mem"


def test_memory_dir_current_name_beats_legacy_name(monkeypatch):
    monkeypatch.delenv("LOCI_MEMORY_MD_DIR", raising=False)
    monkeypatch.setenv("LOCI_MEMORY_DIR", "/current/mem")
    monkeypatch.setenv("HERMES_MEMORY_DIR", "/legacy/mem")
    B._reset_cache()
    assert B.memory_dir() == "/current/mem"
def test_env_overrides_config_for_models(tmp_path, monkeypatch):
    cfg = tmp_path / "b.toml"
    cfg.write_text('[embed]\nmodel="cfg"\n')
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    monkeypatch.setenv("EMBED_MODEL", "envmodel")
    assert B.embed_model() == "envmodel"


def _no_task_model_env(mp):
    for k in ("LOCI_OLLAMA_GEN_MODEL", "LOCI_OLLAMA_VERIFY_MODEL",
              "LOCI_OLLAMA_CLASSIFY_MODEL", "LOCI_OLLAMA_COMPRESS_MODEL"):
        mp.delenv(k, raising=False)


def test_verify_classify_and_compress_model_fall_back_to_gen_model_when_unset(monkeypatch):
    _no_task_model_env(monkeypatch)
    monkeypatch.setattr(B, "_CONFIG_PATH", "/nonexistent")
    monkeypatch.setattr(B, "_ollama_local_tags", lambda: set())
    B._reset_cache()
    assert B.ollama_verify_model() == B.ollama_gen_model()
    assert B.ollama_classify_model() == B.ollama_gen_model()
    assert B.ollama_compress_model() == B.ollama_gen_model()


def test_verify_model_config_key_overrides_gen_model(tmp_path, monkeypatch):
    _no_task_model_env(monkeypatch)
    cfg = tmp_path / "b.toml"
    cfg.write_text('[ollama]\ngen_model = "fast:1b"\nverify_model = "strong:27b"\n')
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    assert B.ollama_gen_model() == "fast:1b"
    assert B.ollama_verify_model() == "strong:27b"
    assert B.ollama_compress_model() == "fast:1b"    # unset -> still falls back


def test_compress_model_env_wins_over_config(tmp_path, monkeypatch):
    _no_task_model_env(monkeypatch)
    cfg = tmp_path / "b.toml"
    cfg.write_text('[ollama]\ncompress_model = "cfg-model:latest"\n')
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    monkeypatch.setenv("LOCI_OLLAMA_COMPRESS_MODEL", "env-model:latest")
    assert B.ollama_compress_model() == "env-model:latest"


def test_classify_model_env_wins_over_config(tmp_path, monkeypatch):
    _no_task_model_env(monkeypatch)
    cfg = tmp_path / "b.toml"
    cfg.write_text('[ollama]\nclassify_model = "cfg-classify:latest"\n')
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    monkeypatch.setenv("LOCI_OLLAMA_CLASSIFY_MODEL", "env-classify:latest")
    assert B.ollama_classify_model() == "env-classify:latest"


def test_guardian_model_defaults_to_granite_guardian_not_gen_model(tmp_path, monkeypatch):
    # Unlike verify/compress, guardian must NOT fall back to a general gen_model:
    # routing a safety classification through an arbitrary chat model would produce
    # meaningless Yes/No output rather than a degraded-but-sane answer.
    monkeypatch.delenv("LOCI_OLLAMA_GUARDIAN_MODEL", raising=False)
    monkeypatch.setattr(B, "_CONFIG_PATH", "/nonexistent")
    monkeypatch.setattr(B, "_ollama_local_tags", lambda: set())
    B._reset_cache()
    monkeypatch.setenv("LOCI_OLLAMA_GEN_MODEL", "some-other-chat-model:latest")
    assert B.ollama_guardian_model() == "granite3-guardian:2b"


def test_guardian_model_config_key_overrides_default(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCI_OLLAMA_GUARDIAN_MODEL", raising=False)
    cfg = tmp_path / "b.toml"
    cfg.write_text('[ollama]\nguardian_model = "granite3-guardian:8b"\n')
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    assert B.ollama_guardian_model() == "granite3-guardian:8b"


def test_guardian_model_env_wins_over_config(tmp_path, monkeypatch):
    cfg = tmp_path / "b.toml"
    cfg.write_text('[ollama]\nguardian_model = "cfg-guardian:latest"\n')
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    monkeypatch.setenv("LOCI_OLLAMA_GUARDIAN_MODEL", "env-guardian:latest")
    assert B.ollama_guardian_model() == "env-guardian:latest"


def test_redteam_model_defaults_to_one_gpu_heretic(monkeypatch):
    monkeypatch.delenv("LOCI_OLLAMA_REDTEAM_MODEL", raising=False)
    monkeypatch.setattr(B, "_CONFIG_PATH", "/nonexistent")
    monkeypatch.setattr(B, "_ollama_local_tags", lambda: set())
    B._reset_cache()
    assert B.ollama_redteam_model() == B.ONE_GPU_REDTEAM_FALLBACK_MODEL == (
        "heretic-llama31-8b-instruct:latest"
    )


# Tags that do not fit one 11-12 GB GPU and must never be picked without env/config.
_OVERSIZED = ("qwen3.8", "gemma4:26b", "27b")


def _assert_one_gpu(model: str) -> None:
    assert not any(marker in model.lower() for marker in _OVERSIZED), model


def _no_swarm_env(mp):
    for k in ("LOCI_SWARM_ESCALATE_MODEL", "LOCI_SWARM_SYNTHESIZE_MODEL",
              "LOCI_OLLAMA_GUARDIAN_MODEL", "LOCI_OLLAMA_REDTEAM_MODEL"):
        mp.delenv(k, raising=False)


def test_no_default_resolves_to_an_oversized_model_without_config(monkeypatch):
    _no_task_model_env(monkeypatch)
    _no_swarm_env(monkeypatch)
    monkeypatch.setattr(B, "_CONFIG_PATH", "/nonexistent")
    monkeypatch.setattr(B, "_ollama_local_tags", lambda: set())
    B._reset_cache()
    for resolver in (B.ollama_gen_model, B.ollama_verify_model, B.ollama_classify_model,
                     B.ollama_compress_model, B.ollama_guardian_model,
                     B.ollama_redteam_model, B.swarm_escalate_model,
                     B.swarm_synthesize_model):
        _assert_one_gpu(resolver())
    assert B.swarm_escalate_model() == B.ONE_GPU_FALLBACK_MODEL
    assert B.swarm_synthesize_model() == B.ONE_GPU_FALLBACK_MODEL


def test_auto_pick_skips_installed_models_too_big_for_one_gpu(monkeypatch):
    # The failure seen 2026-09-26: only oversized tags plus one small one installed.
    _no_task_model_env(monkeypatch)
    _no_swarm_env(monkeypatch)
    monkeypatch.delenv("LOCI_OLLAMA_AUTO_MAX_GB", raising=False)
    monkeypatch.setattr(B, "_CONFIG_PATH", "/nonexistent")
    inventory = {
        "gemma4:26b": 18_600_000_000,
        "qwen3.8:latest": 17_700_000_000,
        "hf.co/slevinw/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:Q4_K_M": 17_200_000_000,
        "tiny-heretic:4b": 3_300_000_000,
        "nomic-embed-text:latest": 274_000_000,
        "small-chat:1b": 1_300_000_000,
    }
    monkeypatch.setattr(B, "_ollama_list", lambda: inventory)
    monkeypatch.setattr(B, "_ollama_local_tags", lambda: set(inventory))
    assert B.ollama_gen_model() == "small-chat:1b"
    assert B.ollama_redteam_model() == "tiny-heretic:4b"
    assert B.ollama_guardian_model() == "small-chat:1b"
    # The cap is an operator knob: raising it lets the big tags back into auto-pick.
    monkeypatch.setenv("LOCI_OLLAMA_AUTO_MAX_GB", "40")
    assert B.ollama_gen_model() == "gemma4:26b"


def test_explicit_config_is_never_size_filtered(tmp_path, monkeypatch):
    _no_task_model_env(monkeypatch)
    _no_swarm_env(monkeypatch)
    cfg = tmp_path / "b.toml"
    cfg.write_text('[ollama]\ngen_model = "gemma4:26b"\nredteam_model = "qwen3.8:latest"\n')
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    monkeypatch.setattr(B, "_ollama_list", lambda: {"gemma4:26b": 18_600_000_000})
    B._reset_cache()
    assert B.ollama_gen_model() == "gemma4:26b"
    assert B.ollama_redteam_model() == "qwen3.8:latest"


def test_swarm_models_resolve_env_then_config_then_verify_model(tmp_path, monkeypatch):
    _no_task_model_env(monkeypatch)
    _no_swarm_env(monkeypatch)
    cfg = tmp_path / "b.toml"
    cfg.write_text('[ollama]\ngen_model = "gemma4-e4b-hermes:64k"\n'
                   'verify_model = "gemma4-e4b-hermes:64k"\n')
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    assert B.swarm_escalate_model() == "gemma4-e4b-hermes:64k"
    assert B.swarm_synthesize_model() == "gemma4-e4b-hermes:64k"

    cfg.write_text('[ollama]\nverify_model = "v:1b"\nswarm_escalate_model = "esc:4b"\n'
                   'swarm_synthesize_model = "syn:4b"\n')
    B._reset_cache()
    assert B.swarm_escalate_model() == "esc:4b"
    assert B.swarm_synthesize_model() == "syn:4b"

    monkeypatch.setenv("LOCI_SWARM_ESCALATE_MODEL", "env-esc:latest")
    monkeypatch.setenv("LOCI_SWARM_SYNTHESIZE_MODEL", "env-syn:latest")
    assert B.swarm_escalate_model() == "env-esc:latest"
    assert B.swarm_synthesize_model() == "env-syn:latest"


def test_ollama_list_parses_sizes(monkeypatch):
    import subprocess

    out = ("NAME                  ID              SIZE      MODIFIED\n"
           "gemma4:26b            abc123          18 GB     2 days ago\n"
           "qwen2.5:3b            def456          1.9 GB    3 weeks ago\n"
           "nomic-embed-text:latest 0a109f422b47  274 MB    5 months ago\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, out, ""))
    monkeypatch.setattr(B, "_ollama_list", _REAL_OLLAMA_LIST)
    B._reset_cache()
    try:
        assert B._ollama_list() == {
            "gemma4:26b": 18_000_000_000,
            "qwen2.5:3b": 1_900_000_000,
            "nomic-embed-text:latest": 274_000_000,
        }
        assert B._ollama_local_tags() == {"gemma4:26b", "qwen2.5:3b", "nomic-embed-text:latest"}
    finally:
        B._reset_cache()


def test_gen_model_auto_selects_local_non_embedding_when_default_missing(monkeypatch):
    _no_task_model_env(monkeypatch)
    monkeypatch.setattr(B, "_CONFIG_PATH", "/nonexistent")
    monkeypatch.setattr(B, "_ollama_local_tags",
                        lambda: {"nomic-embed-text:latest", "local-heretic-llama31-8b:q4km"})
    B._reset_cache()
    assert B.ollama_gen_model() == "local-heretic-llama31-8b:q4km"


def test_guardian_model_auto_selects_local_when_granite_unavailable(monkeypatch):
    monkeypatch.delenv("LOCI_OLLAMA_GUARDIAN_MODEL", raising=False)
    monkeypatch.setattr(B, "_CONFIG_PATH", "/nonexistent")
    monkeypatch.setattr(B, "_ollama_local_tags",
                        lambda: {"nomic-embed-text:latest", "local-heretic-llama31-8b:q4km"})
    B._reset_cache()
    assert B.ollama_guardian_model() == "local-heretic-llama31-8b:q4km"


def test_redteam_model_auto_selects_local_heretic_when_default_tag_unavailable(monkeypatch):
    monkeypatch.delenv("LOCI_OLLAMA_REDTEAM_MODEL", raising=False)
    monkeypatch.setattr(B, "_CONFIG_PATH", "/nonexistent")
    monkeypatch.setattr(B, "_ollama_local_tags", lambda: {"local-heretic-qwen38-27b:q4km"})
    B._reset_cache()
    assert B.ollama_redteam_model() == "local-heretic-qwen38-27b:q4km"


def test_redteam_model_config_key_overrides_default(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCI_OLLAMA_REDTEAM_MODEL", raising=False)
    cfg = tmp_path / "b.toml"
    cfg.write_text('[ollama]\nredteam_model = "heretic-gemma3-4b-it:latest"\n')
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    assert B.ollama_redteam_model() == "heretic-gemma3-4b-it:latest"


def test_redteam_model_env_wins_over_config(tmp_path, monkeypatch):
    cfg = tmp_path / "b.toml"
    cfg.write_text('[ollama]\nredteam_model = "cfg-heretic:latest"\n')
    monkeypatch.setattr(B, "_CONFIG_PATH", str(cfg))
    B._reset_cache()
    monkeypatch.setenv("LOCI_OLLAMA_REDTEAM_MODEL", "env-heretic:latest")
    assert B.ollama_redteam_model() == "env-heretic:latest"


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


# --- Fresh install: all backends together, so the resolution chain is guarded as one unit ---

_ALL_BACKEND_ENV = ("OLLAMA_BASE_URL", "OLLAMA_URL", "EMBED_MODEL",
                    "RERANK_MODEL", "QDRANT_URL", "QDRANT_API_KEY",
                    "LOCI_MEMORY_MD_DIR", "LOCI_MEMORY_DIR")


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
                   '[rerank]\nmodel = "cfg-rerank"\n'
                   '[qdrant]\nurl = "http://cfg-qdrant:6333"\napi_key = "cfg-key"\n'
                   '[memory]\ndir = "/cfg/mem"\n')
    _fresh_install(monkeypatch, cfg)
    assert B.ollama_url() == "http://cfg-gpu:11434"
    assert B.embed_model() == "cfg-embed"
    assert B.rerank_model() == "cfg-rerank"
    assert B.qdrant() == ("http://cfg-qdrant:6333", "cfg-key")
    assert B.memory_dir() == "/cfg/mem"


def test_fresh_install_bare_defaults(monkeypatch):
    _fresh_install(monkeypatch, "/nonexistent/no.toml")
    assert B.ollama_url() == ""                               # url empty -> tiers fail-open
    assert B.embed_model() == "nomic-embed-text"
    assert B.rerank_model() == "BAAI/bge-reranker-v2-m3"      # bge is the flipped fresh default
    assert B.qdrant() == ("", "")
    assert B.memory_dir() == ""
