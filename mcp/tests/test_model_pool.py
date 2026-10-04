"""Model pool: ranked, residency- and size-aware role resolution.

No network and no live store: inventory and residency are injected or monkeypatched.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import backends  # noqa: E402
import model_pool as M  # noqa: E402

GB = 10**9
INV = {
    "gemma4-e4b-hermes:64k": 5 * GB,
    "qwen3:4b-instruct-2507-q8_0": int(4.3 * GB),
    "qwen2.5:3b": int(1.9 * GB),
    "qwen2.5-coder:7b": int(4.7 * GB),
    "gemma4:26b": int(18.6 * GB),
    "nomic-embed-text:latest": int(0.3 * GB),
}
POOL = [
    M.PoolEntry("gemma4-e4b-hermes:64k", ("gen", "verify"), rank=1),
    M.PoolEntry("qwen3:4b-instruct-2507-q8_0", ("gen", "verify"), rank=2),
    M.PoolEntry("qwen2.5:3b", ("gen",), rank=3),
    M.PoolEntry("not-installed:7b", ("gen",), rank=0.5),
    M.PoolEntry("gemma4:26b", ("gen",), rank=0.1),
    M.PoolEntry("qwen2.5-coder:7b", ("code",), rank=1),
]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    M.clear_cache()
    monkeypatch.delenv(M.SHADOW_ENV, raising=False)
    monkeypatch.delenv(M.SELECTOR_ENV, raising=False)
    yield
    M.clear_cache()


def _rank(role, **kw):
    kw.setdefault("pool", POOL)
    kw.setdefault("inv", INV)
    kw.setdefault("resident", set())
    kw.setdefault("bonus", 0.5)
    kw.setdefault("cap_gb", 10.0)
    return M.rank_role(role, **kw)


def test_lowest_rank_eligible_wins_and_skips_missing_and_oversize():
    d = _rank("gen")
    assert d.chosen == "gemma4-e4b-hermes:64k"
    by = {c.name: c for c in d.candidates}
    assert by["not-installed:7b"].eligible is False and by["not-installed:7b"].reason == "not installed"
    assert by["gemma4:26b"].eligible is False and "cap" in by["gemma4:26b"].reason
    assert d.ordered() == ["gemma4-e4b-hermes:64k", "qwen3:4b-instruct-2507-q8_0", "qwen2.5:3b"]


def test_falls_through_when_the_top_rank_is_not_installed():
    inv = {k: v for k, v in INV.items() if k != "gemma4-e4b-hermes:64k"}
    assert _rank("gen", inv=inv).chosen == "qwen3:4b-instruct-2507-q8_0"


def test_a_resident_model_can_overtake_the_next_rank():
    d = _rank("gen", resident={"qwen3:4b-instruct-2507-q8_0"}, bonus=1.5)
    assert d.chosen == "qwen3:4b-instruct-2507-q8_0"      # rank 2 - 1.5 beats rank 1
    assert _rank("gen", resident={"qwen3:4b-instruct-2507-q8_0"}, bonus=0.5).chosen == "gemma4-e4b-hermes:64k"


def test_roles_are_independent():
    assert _rank("code").chosen == "qwen2.5-coder:7b"
    assert _rank("math").chosen == ""


def test_configured_tag_joins_at_rank_zero_but_only_while_installed():
    assert _rank("gen", hint="qwen2.5:3b").chosen == "qwen2.5:3b"             # installed -> wins
    d = _rank("gen", hint="qwen3-4b-instruct-heretic-agent:latest")            # the real incident
    assert d.chosen == "gemma4-e4b-hermes:64k"
    assert any(c.source == "configured" and not c.eligible for c in d.candidates)


def test_tag_without_suffix_matches_latest():
    d = _rank("embed", pool=[M.PoolEntry("nomic-embed-text", ("embed",), 1)])
    assert d.chosen == "nomic-embed-text"


def test_ties_break_deterministically():
    pool = [M.PoolEntry("b", ("gen",), 1), M.PoolEntry("a", ("gen",), 1)]
    inv = {"a": GB, "b": GB}
    assert _rank("gen", pool=pool, inv=inv).chosen == "a"


def test_declared_vram_overrides_the_measured_size_for_the_cap():
    pool = [M.PoolEntry("big", ("gen",), 1, vram_gb=20.0), M.PoolEntry("small", ("gen",), 2)]
    assert _rank("gen", pool=pool, inv={"big": GB, "small": GB}).chosen == "small"


def test_unreachable_inventory_makes_nothing_eligible():
    assert _rank("gen", inv={}).chosen == ""


# ---- configuration -------------------------------------------------------------------

def _cfg(monkeypatch, models):
    monkeypatch.setattr(backends, "_config", lambda: {"models": models} if models is not None else {})


def test_no_pool_configured_is_off(monkeypatch):
    _cfg(monkeypatch, None)
    assert M.configured() is False
    assert M.pick("gen", "anything") == ""
    assert M.summary() == {"configured": False}


def test_entries_parse_and_skip_malformed_rows(monkeypatch):
    _cfg(monkeypatch, {"pool": [
        {"name": "a", "roles": ["gen", "Verify"], "rank": 2, "vram_gb": 4.5, "pinned": True},
        {"name": "b", "roles": "code"},
        {"name": "", "roles": ["gen"]},
        {"name": "c"},
        "junk",
    ]})
    got = M.entries()
    assert [e.name for e in got] == ["a", "b"]
    assert got[0].roles == ("gen", "verify") and got[0].pinned and got[0].vram_gb == 4.5
    assert got[1].roles == ("code",) and got[1].rank == 100.0


def test_pick_uses_injected_inventory_through_the_cache(monkeypatch):
    _cfg(monkeypatch, {"pool": [{"name": "m1", "roles": ["gen"], "rank": 1}]})
    monkeypatch.setattr(M, "inventory", lambda base_url=None: {"m1": GB})
    monkeypatch.setattr(M, "resident_models", lambda base_url=None: set())
    assert M.pick("gen") == "m1"
    assert M.pick("code") == ""            # role not pooled -> legacy resolver


def test_pick_never_raises(monkeypatch):
    _cfg(monkeypatch, {"pool": [{"name": "m1", "roles": ["gen"], "rank": 1}]})

    def boom(*a, **k):
        raise RuntimeError("endpoint exploded")

    monkeypatch.setattr(M, "inventory", boom)
    assert M.pick("gen") == ""


# ---- backends integration ------------------------------------------------------------

def _pooled(monkeypatch, cfg_ollama=None):
    cfg = {"models": {"pool": [
        {"name": "gemma4-e4b-hermes:64k", "roles": ["gen", "verify"], "rank": 1},
        {"name": "qwen3:4b-instruct-2507-q8_0", "roles": ["gen"], "rank": 2},
        {"name": "llama-guard3:8b", "roles": ["guardian"], "rank": 1},
    ]}}
    if cfg_ollama:
        cfg["ollama"] = cfg_ollama
    monkeypatch.setattr(backends, "_config", lambda: cfg)
    monkeypatch.setattr(M, "inventory", lambda base_url=None: dict(INV, **{"llama-guard3:8b": int(4.9 * GB)}))
    monkeypatch.setattr(M, "resident_models", lambda base_url=None: set())
    for var in ("LOCI_OLLAMA_GEN_MODEL", "LOCI_OLLAMA_VERIFY_MODEL", "LOCI_OLLAMA_GUARDIAN_MODEL",
                "LOCI_OLLAMA_REDTEAM_MODEL"):
        monkeypatch.delenv(var, raising=False)


def test_gen_model_falls_through_to_the_pool_when_the_configured_tag_is_missing(monkeypatch):
    _pooled(monkeypatch, {"gen_model": "qwen3-4b-instruct-heretic-agent:latest"})
    assert backends.ollama_gen_model() == "gemma4-e4b-hermes:64k"


def test_a_configured_gen_model_still_wins_while_installed(monkeypatch):
    _pooled(monkeypatch, {"gen_model": "qwen2.5:3b"})
    assert backends.ollama_gen_model() == "qwen2.5:3b"


def test_env_beats_the_pool(monkeypatch):
    _pooled(monkeypatch)
    monkeypatch.setenv("LOCI_OLLAMA_GEN_MODEL", "env-model:1b")
    assert backends.ollama_gen_model() == "env-model:1b"


def test_task_and_guardian_resolvers_use_the_pool(monkeypatch):
    _pooled(monkeypatch, {"verify_model": "missing-verifier:7b"})
    assert backends.ollama_verify_model() == "gemma4-e4b-hermes:64k"
    assert backends.ollama_guardian_model() == "llama-guard3:8b"


def test_unpooled_role_keeps_legacy_behaviour(monkeypatch):
    _pooled(monkeypatch, {"compress_model": "my-compressor:3b"})
    assert backends.ollama_compress_model() == "my-compressor:3b"       # no pooled 'compress' role


def test_without_a_pool_the_legacy_resolvers_are_unchanged(monkeypatch):
    monkeypatch.setattr(backends, "_config", lambda: {"ollama": {"gen_model": "legacy:7b"}})
    monkeypatch.delenv("LOCI_OLLAMA_GEN_MODEL", raising=False)
    assert backends.ollama_gen_model() == "legacy:7b"                   # returned even if not installed


# ---- discovery ----------------------------------------------------------------------

@pytest.mark.parametrize("tag,roles", [
    ("nomic-embed-text:latest", ("embed",)),
    ("llama-guard3:8b", ("guardian", "safety")),
    ("granite3-guardian:2b", ("guardian", "safety")),
    ("qwen2.5-coder:7b", ("code",)),
    ("hf.co/bartowski/Qwen2.5-Math-7B-Instruct-GGUF:Q4_K_M", ("math",)),
    ("hf.co/eaddario/Watt-Tool-8B-GGUF:Q4_K_M", ("tool",)),
    ("minicpm-v:latest", ("vision",)),
    ("MiniCPM-V-4.6-Thinking-heretic:latest", ("vision",)),
    ("heretic-llama31-8b-instruct:latest", ("redteam",)),
    ("gemma4-e4b-hermes:64k-cpu", ("gen_cpu",)),
    ("qwen3:4b-instruct-2507-q8_0", ("gen",)),
])
def test_classify_model_finds_specialists_by_name(tag, roles):
    assert M.classify_model(tag) == roles


def test_suggest_builds_a_draft_pool_without_oversize_models():
    pool = {e.name: e for e in M.suggest(INV, cap_gb=10.0)}
    assert "gemma4:26b" not in pool
    assert pool["qwen2.5-coder:7b"].roles == ("code",)
    assert pool["nomic-embed-text:latest"].pinned is True
    # within 'gen', bigger first
    assert pool["gemma4-e4b-hermes:64k"].rank < pool["qwen2.5:3b"].rank


def test_rendered_toml_round_trips():
    import tomllib
    text = M.render_toml(M.suggest(INV))
    parsed = tomllib.loads(text)
    assert parsed["models"]["pool"] and parsed["models"]["resident_bonus"] == M.DEFAULT_RESIDENT_BONUS
    assert all("name" in row and row["roles"] for row in parsed["models"]["pool"])


# ---- shadow selector ----------------------------------------------------------------

def _shadow_setup(monkeypatch, tmp_path):
    _cfg(monkeypatch, {"pool": [{"name": "a", "roles": ["gen"], "rank": 1},
                                {"name": "b", "roles": ["gen"], "rank": 2}]})
    monkeypatch.setattr(M, "inventory", lambda base_url=None: {"a": GB, "b": GB})
    monkeypatch.setattr(M, "resident_models", lambda base_url=None: set())
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "memory-sessions"))
    return tmp_path / "instrumentation" / M.SHADOW_LOG_NAME


def test_shadow_is_off_by_default(monkeypatch, tmp_path):
    log = _shadow_setup(monkeypatch, tmp_path)
    assert M.pick("gen") == "a"
    assert not log.exists()


def test_shadow_logs_rule_and_selector_choices_without_changing_the_pick(monkeypatch, tmp_path):
    log = _shadow_setup(monkeypatch, tmp_path)
    monkeypatch.setenv(M.SHADOW_ENV, "1")
    monkeypatch.setenv(M.SELECTOR_ENV, "test_model_pool:_prefers_b")
    assert M.pick("gen") == "a"                      # the rule still decides
    row = json.loads(log.read_text().splitlines()[-1])
    assert row["role"] == "gen" and row["chosen_rule"] == "a" and row["chosen_shadow"] == "b"
    assert row["agree"] is False and row["n_eligible"] == 2
    assert set(row) >= {"schema", "ts", "latency_ms", "candidates"}
    blob = json.dumps(row)
    assert "prompt" not in blob and "text" not in row     # names, ranks and enums only


def test_a_failing_selector_is_ignored(monkeypatch, tmp_path):
    log = _shadow_setup(monkeypatch, tmp_path)
    monkeypatch.setenv(M.SHADOW_ENV, "1")
    monkeypatch.setenv(M.SELECTOR_ENV, "test_model_pool:_explodes")
    assert M.pick("gen") == "a"
    row = json.loads(log.read_text().splitlines()[-1])
    assert row["chosen_shadow"] is None and row["agree"] is None


def test_shadow_without_a_selector_still_logs_the_rule_decision(monkeypatch, tmp_path):
    log = _shadow_setup(monkeypatch, tmp_path)
    monkeypatch.setenv(M.SHADOW_ENV, "1")
    assert M.pick("gen") == "a"
    assert json.loads(log.read_text().splitlines()[-1])["chosen_rule"] == "a"


def _prefers_b(role, features):
    return [f["name"] for f in sorted(features, key=lambda f: f["name"], reverse=True)]


def _explodes(role, features):
    raise RuntimeError("selector bug")
