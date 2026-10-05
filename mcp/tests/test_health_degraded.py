"""loci_health must not read ok over a feature that is not working.

Regressions for the 2026-10-05 audit: the findings collection is an ALIAS (loci_memory -> hermes_memory) that
GET /collections does not list, so memory_health and retrieval_selftest reported it missing; embeds and index
writes failed for stretches (investigation_store said stored=true, qdrant_stored=false) while loci_health said
ok; a generation model could sit on CPU unnoticed; the groom script looked at the wrong store and reported
coverage 1.0 over zero findings. Everything here runs in-memory or against stubs.
"""
import importlib
import json
import os
import sys

import pytest

qdrant_client = pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient, models  # noqa: E402

import backends  # noqa: E402
import qdrant_ops  # noqa: E402
import server  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_health_state():
    for events in qdrant_ops._health_events.values():
        events.clear()
    server._main_state_cache.update(t=0.0, v=None)
    yield
    for events in qdrant_ops._health_events.values():
        events.clear()
    server._main_state_cache.update(t=0.0, v=None)


def _client_with_alias():
    c = QdrantClient(":memory:")
    c.create_collection("hermes_memory", vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)})
    c.update_collection_aliases(change_aliases_operations=[models.CreateAliasOperation(
        create_alias=models.CreateAlias(collection_name="hermes_memory", alias_name="loci_memory"))])
    return c


# ----------------------------------------------------------------------------------------- aliases

def test_the_plain_collection_list_hides_aliases_and_the_helper_does_not():
    c = _client_with_alias()
    assert "loci_memory" not in {x.name for x in c.get_collections().collections}   # the trap
    names, aliases = qdrant_ops.collection_names_with_aliases(c)
    assert {"hermes_memory", "loci_memory"} <= names
    assert aliases == {"loci_memory": "hermes_memory"}


def test_memory_health_collection_probe_resolves_the_alias():
    status, report, hint = server._health_probe_qdrant_collections(_client_with_alias(), "loci_memory", {})
    assert report["main_present"] is True
    assert report["main_is_alias_of"] == "hermes_memory"
    assert status != "fail", (status, hint)


def test_memory_health_still_fails_when_the_main_name_resolves_to_nothing():
    c = QdrantClient(":memory:")
    status, report, _ = server._health_probe_qdrant_collections(c, "loci_memory", {})
    assert status == "fail" and report["main_present"] is False


def test_retrieval_selftest_does_not_call_an_aliased_collection_missing(monkeypatch):
    c = _client_with_alias()
    monkeypatch.setattr(server, "_qdrant_client_readonly", lambda: (c, "loci_memory"))
    monkeypatch.setattr(server, "_embed_uncached", lambda text: [0.1, 0.1, 0.1, 0.1])
    out = json.loads(server.retrieval_selftest(scope="queried"))
    rows = {r["collection"]: r for r in out["collections"]}
    assert rows["loci_memory"]["status"] != "missing", out


# ------------------------------------------------------------------------------ rolling transport health

def test_embed_failures_are_counted_and_a_success_ends_the_run():
    for ok in (True, False, False, False):
        qdrant_ops._health_note("embed", ok)
    h = qdrant_ops.transport_health()["embed"]
    assert h["consecutive_failures"] == 3 and h["failed"] == 3 and h["ok"] == 1
    assert h["last_ok_age_s"] is not None and h["last_failure_age_s"] is not None
    qdrant_ops._health_note("embed", True)
    assert qdrant_ops.transport_health()["embed"]["consecutive_failures"] == 0


def test_nothing_recorded_means_nothing_reported():
    h = qdrant_ops.transport_health()
    assert h["embed"]["consecutive_failures"] == 0 and h["embed"]["last_ok_age_s"] is None
    assert h["index_write"]["failed"] == 0


def test_a_failed_embed_is_recorded(monkeypatch):
    monkeypatch.setattr(qdrant_ops, "_embed_cache", {})
    monkeypatch.setattr(qdrant_ops, "_endpoint_ready_cache", {})
    monkeypatch.setattr(qdrant_ops, "_transport_breakers", {})
    monkeypatch.setattr(qdrant_ops, "_OLLAMA_BASE", "http://127.0.0.1:9")   # nothing listens on the discard port
    assert qdrant_ops._embed("anything", use_cache=False) is None
    assert qdrant_ops.transport_health()["embed"]["consecutive_failures"] == 1


def test_a_failed_index_write_is_recorded(monkeypatch):
    class _C:
        def upsert(self, *a, **k):
            raise RuntimeError("boom")
    monkeypatch.setattr(qdrant_ops, "_get_qdrant", lambda: (_C(), "loci_memory"))
    monkeypatch.setattr(qdrant_ops, "_embed", lambda text, use_cache=True: [0.1, 0.2])
    monkeypatch.setattr(qdrant_ops, "_embed_sparse", lambda text: None)
    assert qdrant_ops._qdrant_upsert("id1", "text", {}) is False
    assert qdrant_ops.transport_health()["index_write"]["consecutive_failures"] == 1


def test_a_missing_vector_counts_as_a_failed_index_write(monkeypatch):
    monkeypatch.setattr(qdrant_ops, "_get_qdrant", lambda: (object(), "loci_memory"))
    monkeypatch.setattr(qdrant_ops, "_embed", lambda text, use_cache=True: None)
    assert qdrant_ops._qdrant_upsert("id1", "text", {}) is False
    assert qdrant_ops.transport_health()["index_write"]["consecutive_failures"] == 1


# --------------------------------------------------------------------------------- the assessment

def _t(kind, consec=0, failed=0, **extra):
    base = {"consecutive_failures": consec, "failed": failed, "window_s": 900, "last_ok_age_s": 42.0}
    base.update(extra)
    return {kind: base}


def test_three_consecutive_embed_failures_degrade_two_do_not():
    assert server._assess_degradation(_t("embed", consec=2, failed=2), None, None)[0] == []
    reasons, _ = server._assess_degradation(_t("embed", consec=3, failed=3, breaker_open_s=12.0), None, None)
    assert len(reasons) == 1 and "embedding" in reasons[0] and "3 consecutive" in reasons[0] and "breaker" in reasons[0]


def test_one_failed_index_write_right_now_degrades_and_says_what_it_means():
    reasons, _ = server._assess_degradation(_t("index_write", consec=1, failed=1), None, None)
    assert reasons and "qdrant_stored=false" in reasons[0]


def test_a_recovered_failure_is_a_warning_not_a_degradation():
    reasons, warnings = server._assess_degradation(_t("index_write", consec=0, failed=2), None, None)
    assert reasons == [] and any("recovered" in w for w in warnings)


def test_main_collection_not_resolving_degrades():
    reasons, _ = server._assess_degradation(None, {"name": "loci_memory", "present": False}, None)
    assert reasons and "does not resolve" in reasons[0]
    assert server._assess_degradation(None, {"name": "loci_memory", "present": True, "alias_of": "hermes_memory"}, None)[0] == []


def test_a_generation_model_on_cpu_degrades_one_on_gpu_does_not():
    cpu = {"model": "m:16k", "resident": True, "on_gpu": False}
    assert "CPU" in server._assess_degradation(None, None, cpu)[0][0]
    assert server._assess_degradation(None, None, {"model": "m", "resident": True, "on_gpu": True}) == ([], [])
    reasons, warnings = server._assess_degradation(None, None, {"model": "m", "resident": False, "on_gpu": None})
    assert reasons == [] and warnings


def test_d10_swallowed_errors_are_a_warning():
    reasons, warnings = server._assess_degradation(None, None, None, d10_errors=3)
    assert reasons == [] and any("D10" in w for w in warnings)


def test_unmeasured_inputs_never_degrade():
    assert server._assess_degradation(None, None, None) == ([], [])


# --------------------------------------------------------------------------------- gen residency

def _ps(models_):
    return lambda url, path="", timeout=1.0, headers=None: (True, {"models": models_})


def test_gen_residency_reads_size_vram(monkeypatch):
    gb = 10**9
    monkeypatch.setattr(backends, "_http_probe", _ps([{"name": "g:16k", "size": 5 * gb, "size_vram": 5 * gb}]))
    r = server._gen_residency("http://x", "g:16k")
    assert r["resident"] and r["on_gpu"] is True and r["size_vram_gb"] == 5.0
    monkeypatch.setattr(backends, "_http_probe", _ps([{"name": "g:16k", "size": 5 * gb, "size_vram": 0}]))
    assert server._gen_residency("http://x", "g:16k")["on_gpu"] is False
    monkeypatch.setattr(backends, "_http_probe", _ps([{"name": "other", "size": gb, "size_vram": gb}]))
    r = server._gen_residency("http://x", "g:16k")
    assert r["resident"] is False and r["on_gpu"] is None
    monkeypatch.setattr(backends, "_http_probe", lambda *a, **k: (False, None))
    assert server._gen_residency("http://x", "g:16k") is None


# ------------------------------------------------------------------------------------ loci_health

def _everything_reachable(monkeypatch):
    """Make every backend probe in loci_health answer, so the baseline status is ok."""
    monkeypatch.setattr(backends, "_alive", lambda *a, **k: True)
    monkeypatch.setattr(backends, "_http_probe", lambda *a, **k: (True, {}))
    monkeypatch.setattr(server, "_ladybug_health_state", lambda: "available")


def test_loci_health_reports_degraded_with_reasons_when_embeds_fail(monkeypatch):
    _everything_reachable(monkeypatch)
    monkeypatch.setattr(qdrant_ops, "transport_health",
                        lambda *a, **k: {**_t("embed", consec=5, failed=5), **_t("index_write", consec=2, failed=2)})
    out = json.loads(server.loci_health())
    assert out["status"] == "degraded", out
    assert any("embedding" in r for r in out["degraded_reasons"])
    assert any("index writes" in r for r in out["degraded_reasons"])
    assert out["embed_health"]["consecutive_failures"] == 5


def test_loci_health_stays_ok_when_nothing_is_failing(monkeypatch):
    _everything_reachable(monkeypatch)
    out = json.loads(server.loci_health())
    assert out["status"] == "ok", out
    assert "degraded_reasons" not in out


def test_unhealthy_is_not_downgraded_to_degraded(monkeypatch):
    monkeypatch.setattr(qdrant_ops, "transport_health", lambda *a, **k: _t("embed", consec=9, failed=9))
    monkeypatch.setattr(server, "_ladybug_health_state", lambda: "latched")
    out = json.loads(server.loci_health())
    assert out["status"] == "unhealthy" and out.get("degraded_reasons")


# ------------------------------------------------------------------------------------ data home

def _load_groom(name):
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "loci_groom.py")
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_groom_uses_the_servers_data_home_rule(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "mem"))
    groom = _load_groom("loci_groom_under_test_a")
    assert str(groom.MEMORY_DIR) == str(tmp_path / "mem")


def test_groom_default_matches_legacy_env_without_the_override(monkeypatch):
    monkeypatch.delenv("LOCI_MEMORY_DIR", raising=False)
    monkeypatch.delenv("HERMES_MEMORY_DIR", raising=False)
    import legacy_env
    groom = _load_groom("loci_groom_under_test_b")
    assert str(groom.MEMORY_DIR) == str(legacy_env.memory_dir())


def test_a_failing_assessment_is_a_warning_not_silence(monkeypatch):
    _everything_reachable(monkeypatch)
    def boom(*a, **k):
        raise RuntimeError("probe broke")
    monkeypatch.setattr(qdrant_ops, "transport_health", boom)
    out = json.loads(server.loci_health())
    assert any("assessment failed" in w and "probe broke" in w for w in out.get("warnings", [])), out
