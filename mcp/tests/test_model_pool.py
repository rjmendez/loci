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


def test_a_named_gen_model_in_config_is_ignored_missing_or_installed(monkeypatch):
    """The pool decides; a config tag no longer outranks it, whether or not it is installed."""
    _pooled(monkeypatch, {"gen_model": "qwen3-4b-instruct-heretic-agent:latest"})
    assert backends.ollama_gen_model() == "gemma4-e4b-hermes:64k"
    _pooled(monkeypatch, {"gen_model": "qwen2.5:3b"})
    assert backends.ollama_gen_model() == "gemma4-e4b-hermes:64k"


def test_env_beats_the_pool(monkeypatch):
    _pooled(monkeypatch)
    monkeypatch.setenv("LOCI_OLLAMA_GEN_MODEL", "env-model:1b")
    assert backends.ollama_gen_model() == "env-model:1b"


def test_task_and_guardian_resolvers_use_the_pool(monkeypatch):
    _pooled(monkeypatch, {"verify_model": "missing-verifier:7b"})
    assert backends.ollama_verify_model() == "gemma4-e4b-hermes:64k"
    assert backends.ollama_guardian_model() == "llama-guard3:8b"


def test_without_a_pool_the_gen_model_is_an_installed_one_never_a_config_name(monkeypatch):
    monkeypatch.setattr(backends, "_config", lambda: {"ollama": {"gen_model": "legacy:7b"}})
    monkeypatch.setattr(backends, "_ollama_local_tags", lambda: {"nomic-embed-text:latest", "installed:3b"})
    monkeypatch.delenv("LOCI_OLLAMA_GEN_MODEL", raising=False)
    assert backends.ollama_gen_model() == "installed:3b"


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


# ---- over-cap fallback and per-role ranks --------------------------------------------

CAP_INV = {
    "small-verifier:8b": int(4.9 * GB),
    "big-verifier:26b": int(18.6 * GB),
    "bigger-verifier:27b": int(17.2 * GB),
}
CAP_POOL = [
    M.PoolEntry("small-verifier:8b", ("verify",), rank=1),
    M.PoolEntry("big-verifier:26b", ("verify",), rank=10),
    M.PoolEntry("bigger-verifier:27b", ("verify",), rank=11),
]


def test_cap_prefers_a_model_that_fits_even_when_a_bigger_one_ranks_higher():
    pool = [M.PoolEntry("big-verifier:26b", ("verify",), rank=1),
            M.PoolEntry("small-verifier:8b", ("verify",), rank=2)]
    d = _rank("verify", pool=pool, inv=CAP_INV, fallback=True)
    assert d.chosen == "small-verifier:8b"
    assert not any(c.over_cap for c in d.candidates)


def test_without_fallback_the_cap_leaves_the_role_unresolved():
    inv = {k: v for k, v in CAP_INV.items() if k != "small-verifier:8b"}
    assert _rank("verify", pool=CAP_POOL, inv=inv, fallback=False).chosen == ""


def test_when_nothing_fits_the_strongest_over_cap_model_is_used():
    inv = {k: v for k, v in CAP_INV.items() if k != "small-verifier:8b"}
    d = _rank("verify", pool=CAP_POOL, inv=inv, fallback=True)
    assert d.chosen == "big-verifier:26b"                  # best rank among the over-cap
    chosen = next(c for c in d.candidates if c.name == d.chosen)
    assert chosen.over_cap and "nothing fits" in chosen.reason
    assert d.ordered() == ["big-verifier:26b", "bigger-verifier:27b"]


def test_fallback_never_resurrects_a_model_that_is_not_installed():
    d = _rank("verify", pool=CAP_POOL, inv={}, fallback=True)
    assert d.chosen == ""


def test_the_cap_relaxes_per_role_not_globally():
    pool = CAP_POOL + [M.PoolEntry("small-gen:3b", ("gen",), rank=1)]
    inv = dict(CAP_INV, **{"small-gen:3b": 2 * GB})
    inv.pop("small-verifier:8b")
    assert _rank("verify", pool=pool, inv=inv, fallback=True).chosen == "big-verifier:26b"
    assert _rank("gen", pool=pool, inv=inv, fallback=True).chosen == "small-gen:3b"


def test_role_rank_overrides_the_entry_rank_for_that_role_only():
    pool = [M.PoolEntry("a", ("gen", "verify"), rank=1, role_rank=(("verify", 5.0),)),
            M.PoolEntry("b", ("gen", "verify"), rank=2)]
    inv = {"a": GB, "b": GB}
    assert _rank("gen", pool=pool, inv=inv).chosen == "a"
    assert _rank("verify", pool=pool, inv=inv).chosen == "b"


def test_configured_hint_uses_rank_zero_even_with_a_role_rank():
    pool = [M.PoolEntry("a", ("verify",), rank=1, role_rank=(("verify", 9.0),)),
            M.PoolEntry("b", ("verify",), rank=2)]
    assert _rank("verify", pool=pool, inv={"a": GB, "b": GB}, hint="a").chosen == "a"


def test_role_rank_and_over_cap_fallback_parse_from_config(monkeypatch):
    _cfg(monkeypatch, {"over_cap_fallback": True, "max_vram_gb": 10,
                       "pool": [{"name": "m", "roles": ["verify"], "rank": 1,
                                 "role_rank": {"Verify": 3, "gen": 7}}]})
    assert M.over_cap_fallback() is True and M.max_vram_gb() == 10.0
    (entry,) = M.entries()
    assert entry.rank_for("verify") == 3.0 and entry.rank_for("gen") == 7.0 and entry.rank_for("code") == 1.0


def test_over_cap_fallback_defaults_off(monkeypatch):
    _cfg(monkeypatch, {"pool": [{"name": "m", "roles": ["gen"]}]})
    assert M.over_cap_fallback() is False


def test_summary_marks_an_over_cap_pick(monkeypatch):
    _cfg(monkeypatch, {"max_vram_gb": 10, "over_cap_fallback": True,
                       "pool": [{"name": "big-verifier:26b", "roles": ["verify"], "rank": 1}]})
    monkeypatch.setattr(M, "inventory", lambda base_url=None: {"big-verifier:26b": int(18.6 * GB)})
    monkeypatch.setattr(M, "resident_models", lambda base_url=None: set())
    info = M.summary()["roles"]["verify"]
    assert info["chosen"] == "big-verifier:26b" and info["over_cap"] is True


def test_rendered_toml_keeps_role_rank():
    import tomllib
    text = M.render_toml([M.PoolEntry("m", ("gen", "verify"), 1, role_rank=(("verify", 4.0),))])
    row = tomllib.loads(text)["models"]["pool"][0]
    assert row["role_rank"] == {"verify": 4.0}


# ---- stale-if-error inventory cache ---------------------------------------------------

def _fake_http(monkeypatch, responses):
    calls = []

    def fake(url):
        calls.append(url)
        item = responses.pop(0) if responses else None
        return item

    monkeypatch.setattr(M, "_get_json", fake)
    return calls


def test_a_failed_refresh_keeps_serving_the_last_good_inventory(monkeypatch):
    calls = _fake_http(monkeypatch, [{"models": [{"name": "m1", "size": GB}]}, None])
    assert M.inventory("http://x") == {"m1": GB}
    monkeypatch.setattr(M, "_TAGS_TTL_S", 0.0)          # force a refresh on the next call
    assert M.inventory("http://x") == {"m1": GB}        # refresh failed -> stale answer, not {}
    assert len(calls) == 2


def test_first_ever_failure_is_empty_and_cached_for_the_ttl(monkeypatch):
    calls = _fake_http(monkeypatch, [None, {"models": [{"name": "m1", "size": GB}]}])
    assert M.inventory("http://y") == {}
    assert M.inventory("http://y") == {}                # cached failure: the endpoint is not hammered
    assert len(calls) == 1


def test_stale_answers_expire(monkeypatch):
    _fake_http(monkeypatch, [{"models": [{"name": "m1", "size": GB}]}, None])
    assert M.inventory("http://z") == {"m1": GB}
    monkeypatch.setattr(M, "_TAGS_TTL_S", 0.0)
    monkeypatch.setattr(M, "_STALE_MAX_S", 0.0)
    assert M.inventory("http://z") == {}


def test_residency_also_survives_a_failed_refresh(monkeypatch):
    _fake_http(monkeypatch, [{"models": [{"name": "m1"}]}, None])
    assert M.resident_models("http://r") == {"m1"}
    monkeypatch.setattr(M, "_PS_TTL_S", 0.0)
    assert M.resident_models("http://r") == {"m1"}


# ---- outcome logging (labels for the pool decision) -----------------------------------

def _outcomes_log(tmp_path):
    return tmp_path / "instrumentation" / M.OUTCOMES_LOG_NAME


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_outcomes_are_not_logged_by_default(monkeypatch, tmp_path):
    import llm_local
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "memory-sessions"))
    monkeypatch.setattr(llm_local, "_generate", lambda *a, **k: {"text": "hi", "ok": True, "model": "m1"})
    assert llm_local.generate("hello")["ok"] is True
    assert not _outcomes_log(tmp_path).exists()


def test_generate_logs_model_ok_latency_and_enums_only(monkeypatch, tmp_path):
    import llm_local
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "memory-sessions"))
    monkeypatch.setenv(M.SHADOW_ENV, "1")
    monkeypatch.setattr(llm_local, "_generate", lambda *a, **k: {"text": "SECRET OUTPUT", "ok": True, "model": "m1"})
    out = llm_local.generate("SECRET PROMPT", fmt="json", role="triage")
    assert out["text"] == "SECRET OUTPUT"                      # the caller's result is untouched
    (row,) = _rows(_outcomes_log(tmp_path))
    assert row["model"] == "m1" and row["ok"] is True and row["fmt"] == "json" and row["route_role"] == "triage"
    assert row["tier"] == "ollama" and row["deadline_exceeded"] is False and row["latency_ms"] >= 0
    blob = json.dumps(row)
    assert "SECRET" not in blob and "why" not in row and "text" not in row


def test_a_failed_call_and_a_deadline_are_logged_as_such(monkeypatch, tmp_path):
    import llm_local
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "memory-sessions"))
    monkeypatch.setenv(M.SHADOW_ENV, "1")
    monkeypatch.setattr(llm_local, "_generate", lambda *a, **k: {
        "text": "", "ok": False, "model": "m2", "why": "boom: private detail", "deadline_exceeded": True})
    llm_local.generate("p", model="m2")
    (row,) = _rows(_outcomes_log(tmp_path))
    assert row["ok"] is False and row["deadline_exceeded"] is True and row["model"] == "m2"
    assert "private detail" not in json.dumps(row)


def test_the_requested_model_is_logged_when_the_result_names_none(monkeypatch, tmp_path):
    import llm_local
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "memory-sessions"))
    monkeypatch.setenv(M.SHADOW_ENV, "1")
    monkeypatch.setattr(llm_local, "_generate", lambda *a, **k: {"text": "", "ok": False})
    llm_local.generate("p", model="asked-for:7b")
    assert _rows(_outcomes_log(tmp_path))[0]["model"] == "asked-for:7b"


def test_a_logging_failure_never_breaks_generate(monkeypatch, tmp_path):
    import llm_local
    monkeypatch.setenv(M.SHADOW_ENV, "1")
    monkeypatch.setattr(llm_local, "_generate", lambda *a, **k: {"text": "hi", "ok": True, "model": "m1"})

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(M, "record_outcome", boom)
    assert llm_local.generate("p") == {"text": "hi", "ok": True, "model": "m1"}


def test_generate_keeps_its_signature_and_docstring():
    import inspect
    import llm_local
    names = list(inspect.signature(llm_local.generate).parameters)
    assert names == ["prompt", "model", "fmt", "max_tokens", "temperature", "keep_alive", "think",
                     "role", "timeout"]
    assert "Fail-open, never raises" in (llm_local.generate.__doc__ or "")


def test_outcomes_summary_aggregates_per_model(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "memory-sessions"))
    monkeypatch.setenv(M.SHADOW_ENV, "1")
    for ok, ms, dl in ((True, 100, False), (True, 300, False), (False, 900, True), (True, 200, False)):
        M.record_outcome("m1", ok, ms, deadline_exceeded=dl)
    M.record_outcome("m2", True, 50)
    s = M.outcomes_summary()
    assert s["m1"]["calls"] == 4 and s["m1"]["ok_rate"] == 0.75 and s["m1"]["deadline_exceeded"] == 1
    assert s["m1"]["p95_ms"] == 900.0 and s["m1"]["p50_ms"] == 300.0
    assert s["m2"] == {"calls": 1, "ok_rate": 1.0, "p50_ms": 50.0, "p95_ms": 50.0, "deadline_exceeded": 0}


def test_outcomes_summary_skips_bad_lines_and_a_missing_log(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "memory-sessions"))
    assert M.outcomes_summary() == {}
    path = _outcomes_log(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('not json\n{"model": "m1", "ok": true, "latency_ms": 5}\n{"no_model": 1}\n')
    assert M.outcomes_summary() == {"m1": {"calls": 1, "ok_rate": 1.0, "p50_ms": 5.0, "p95_ms": 5.0,
                                           "deadline_exceeded": 0}}


def test_outcomes_cli_prints_a_row_per_model(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "memory-sessions"))
    monkeypatch.setenv(M.SHADOW_ENV, "1")
    M.record_outcome("m1", True, 120)
    assert M._main(["outcomes"]) == 0
    out = capsys.readouterr().out
    assert "m1" in out and "ok=100%" in out
