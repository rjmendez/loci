"""The D10 pair gate runs in shadow only: it is scored and logged, and never decides.

Covers the artifact (numpy inference equals sklearn; any altered byte refuses to load)
and the investigation_reason hook (off by default and then does nothing; on, it logs
ids and scores but no text and leaves the live keep/drop exactly as it was; any failure
in it is swallowed and counted).
"""
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import d10_gate
import memcheck.llm as llm
import server

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "deep_think_loci" / "grounding" / "grounding_dataset.jsonl"


def _trainer():
    spec = importlib.util.spec_from_file_location(
        "train_d10_pair_gate", REPO / "deep_think_loci" / "grounding" / "train_d10_pair_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- artifact


def _toy_export(tmp_path, embed_dim=8, n=240):
    """Fit the real recipe on synthetic pairs at a small width and export it."""
    T = _trainer()
    rng = np.random.default_rng(7)
    u = rng.normal(size=(n, embed_dim))
    y = rng.integers(0, 2, size=n)
    v = np.where(y[:, None] == 1, u + rng.normal(scale=0.5, size=u.shape), rng.normal(size=u.shape))
    ta = [f"alpha {i} not ready" for i in range(n)]
    tb = [f"beta {i % 9} value 3" for i in range(n)]
    X = d10_gate.pair_features(u, v, ta, tb)
    model = T.fit(X, y)
    meta = {"embed_dim": embed_dim, "recipe": dict(T.RECIPE),
            "threshold": {"name": "tau_R95", "tau": 0.5, "recall_floor": 0.95}}
    out = tmp_path / "artifact"
    msha, _ = T.export(model, out, meta)
    return model, X, out, msha


def test_numpy_inference_equals_the_sklearn_model_on_a_fixed_batch(tmp_path):
    model, X, out, msha = _toy_export(tmp_path)
    gate = d10_gate.load_gate(out, expected_manifest_sha256=msha)
    batch = X[:64]
    np.testing.assert_allclose(gate.score(batch), model.predict_proba(batch)[:, 1], rtol=0, atol=1e-6)


def test_the_committed_artifact_reproduces_the_sklearn_probabilities_it_recorded():
    gate = d10_gate.load_gate()
    manifest = json.loads((d10_gate.ARTIFACT_DIR / d10_gate.MANIFEST_NAME).read_text())
    Xc = _trainer().check_batch(gate.n_features, manifest["embed_dim"])
    np.testing.assert_allclose(gate.score(Xc), manifest["check"]["sklearn_proba"], rtol=0, atol=1e-6)
    assert gate.n_features == 4 * 768 + 10
    assert manifest["n_params"] == 203541


def test_the_committed_manifest_names_its_data_recipe_threshold_and_caveats():
    manifest = json.loads((d10_gate.ARTIFACT_DIR / d10_gate.MANIFEST_NAME).read_text())
    assert manifest["training_data"]["sha256"] == hashlib.sha256(DATASET.read_bytes()).hexdigest()
    assert manifest["training_data"]["loci_commit"].startswith("113efecb")
    assert manifest["recipe"]["hidden"] == 64 and manifest["recipe"]["alpha"] == 1e-3
    assert manifest["threshold"]["name"] == "tau_R95" and 0 < manifest["threshold"]["tau"] < 1
    assert round(manifest["loro_reproduced_here"]["mlp_m1_tnr_at_r95"], 3) == 0.987
    assert round(manifest["loro_reproduced_here"]["cosine_m1_tnr_at_r95"], 3) == 0.949
    assert any("structural proxies" in c for c in manifest["loro_offline_results"]["caveats"])


def _copy_artifact(tmp_path):
    dst = tmp_path / "copy"
    shutil.copytree(d10_gate.ARTIFACT_DIR, dst)
    return dst


def test_an_altered_weights_file_refuses_to_load(tmp_path):
    d = _copy_artifact(tmp_path)
    w = d / d10_gate.WEIGHTS_NAME
    data = bytearray(w.read_bytes())
    data[len(data) // 2] ^= 0x01
    w.write_bytes(bytes(data))
    with pytest.raises(d10_gate.D10GateError, match="does not match the manifest"):
        d10_gate.load_gate(d)


def test_an_altered_manifest_refuses_to_load_against_the_pin(tmp_path):
    d = _copy_artifact(tmp_path)
    m = d / d10_gate.MANIFEST_NAME
    manifest = json.loads(m.read_text())
    manifest["threshold"]["tau"] = 0.0          # a looser gate, weights untouched
    m.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    with pytest.raises(d10_gate.D10GateError, match="pinned"):
        d10_gate.load_gate(d)


def test_the_pin_is_the_committed_manifest():
    raw = (d10_gate.ARTIFACT_DIR / d10_gate.MANIFEST_NAME).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == d10_gate.PINNED_MANIFEST_SHA256


def test_a_missing_artifact_refuses_to_load(tmp_path):
    with pytest.raises(d10_gate.D10GateError, match="no D10 artifact"):
        d10_gate.load_gate(tmp_path)


def test_pair_features_layout():
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=(3, 5)), rng.normal(size=(3, 5))
    X = d10_gate.pair_features(u, v, ["not 1 a", "x", ""], ["1 a 2", "x", ""])
    assert X.shape == (3, 4 * 5 + 10)
    un = u / np.linalg.norm(u, axis=1, keepdims=True)
    vn = v / np.linalg.norm(v, axis=1, keepdims=True)
    np.testing.assert_allclose(X[:, :5], un, atol=1e-8)
    np.testing.assert_allclose(X[:, 20], np.sum(un * vn, axis=1), atol=1e-8)
    np.testing.assert_allclose(X[:, 21], X[:, 20] ** 2)
    # "not 1 a" vs "1 a 2": jaccard 2/4, one side negated, one shared number, one only-one
    assert X[0, 22] == pytest.approx(0.5)
    assert X[0, 26:30].tolist() == pytest.approx([1.0, 1.0, np.log1p(1), np.log1p(1)])
    assert X[2, 22] == 1.0      # two empty texts: jaccard defined as 1


# --------------------------------------------------------------------------- shadow hook

QUESTION = "Why does the payment worker exceed its memory limit?"
FINDINGS = [
    "payment worker heap grows 40 MB per batch until the OOM killer fires",
    "the payment worker retains every parsed invoice in a module-level list",
    "unrelated: the docs site uses a dark theme",
    "the CI runner image was bumped to Ubuntu 24.04",
    "memory limit for the payment worker pod is 512 MiB",
]
ON_TOPIC = {0, 1, 4}
LOG_FIELDS = {"schema", "event", "ts", "call_id", "investigation_id", "finding_id", "n_pairs", "n_logged",
              "cos", "cos_threshold", "cos_keep", "cos_rank", "cos_in_context", "mlp_score", "mlp_tau",
              "mlp_keep", "mlp_in_context", "gate_version", "latency_us_call", "latency_us_per_pair",
              "shadow_errors_total"}


@pytest.fixture
def memory(tmp_path, monkeypatch):
    mem = tmp_path / "memory-sessions"
    mem.mkdir()
    original = server.MEMORY_DIR
    server.MEMORY_DIR = mem
    server._session_hints.clear()
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(mem))
    monkeypatch.setattr(d10_gate, "_cached", None)
    yield mem
    server.MEMORY_DIR = original
    server._session_hints.clear()


def _fake_embed(texts, **kwargs):
    base = np.random.default_rng(1).normal(size=768)
    out = []
    for t in texts:
        seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
        noise = np.random.default_rng(seed).normal(size=768)
        idx = FINDINGS.index(t) if t in FINDINGS else None
        on = t == QUESTION or idx in ON_TOPIC
        out.append((base + (0.4 if on else 3.0) * noise).tolist())
    return out


def _seed(inv_id):
    server.investigation_start(investigation_id=inv_id, title="d10 shadow test")
    ids = []
    for text in FINDINGS:
        stored = json.loads(server.investigation_store(
            investigation_id=inv_id, finding_type="observed", text=text, source="unit-test", confidence="high"))
        ids.append(stored["finding_id"])
    return ids


def _run(monkeypatch, inv_id):
    prompts = []

    def _fake_call(prompt, **kwargs):
        prompts.append(prompt)
        if kwargs.get("json_mode"):
            return json.dumps({"converged_claims": [], "contested_areas": [], "final_answer": "done",
                               "confidence_score": 50})
        return "analysis"

    monkeypatch.setattr(llm, "llm_available", lambda: True)
    monkeypatch.setattr(llm, "embed_texts", _fake_embed)
    monkeypatch.setattr(llm, "call_llm", _fake_call)
    result = json.loads(server.investigation_reason(inv_id, QUESTION, perspectives=1, persist=False))
    return result, prompts


def _shadow_log(memory):
    return memory.parent / "instrumentation" / d10_gate.SHADOW_LOG_NAME


def _live_decision(result, prompts):
    """What the model was shown: the gate's count and the evidence block of the prompt."""
    return result["grounded_findings"], result["gate_applied"], prompts[0].split("GROUNDED EVIDENCE", 1)[1]


def _boom(*args, **kwargs):
    raise AssertionError("the shadow path ran while LOCI_D10_SHADOW was off")


def test_shadow_off_by_default_does_no_work_and_writes_no_file(memory, monkeypatch):
    monkeypatch.delenv("LOCI_D10_SHADOW", raising=False)
    _seed("d10-off")
    monkeypatch.setattr(d10_gate, "record_shadow", _boom)
    monkeypatch.setattr(d10_gate, "load_gate", _boom)
    monkeypatch.setattr(d10_gate, "pair_features", _boom)
    result, _ = _run(monkeypatch, "d10-off")
    assert result["gate_applied"] is True
    assert not (memory.parent / "instrumentation").exists()
    assert not list(memory.parent.rglob("d10_shadow*"))


def test_shadow_off_when_explicitly_zero(memory, monkeypatch):
    monkeypatch.setenv("LOCI_D10_SHADOW", "0")
    _seed("d10-zero")
    monkeypatch.setattr(d10_gate, "record_shadow", _boom)
    _run(monkeypatch, "d10-zero")
    assert not _shadow_log(memory).exists()


def test_shadow_on_logs_every_pair_and_never_changes_the_live_decision(memory, monkeypatch):
    ids = _seed("d10-on")
    monkeypatch.delenv("LOCI_D10_SHADOW", raising=False)
    off = _live_decision(*_run(monkeypatch, "d10-on"))
    monkeypatch.setenv("LOCI_D10_SHADOW", "1")
    on = _live_decision(*_run(monkeypatch, "d10-on"))
    assert on == off
    assert off[0] == len(ON_TOPIC)          # the fixture really exercises keep and drop

    rows = [json.loads(line) for line in _shadow_log(memory).read_text().splitlines()]
    assert len(rows) == len(FINDINGS)
    assert {r["finding_id"] for r in rows} == set(ids)
    assert all(set(r) == LOG_FIELDS for r in rows)
    assert len({r["call_id"] for r in rows}) == 1
    assert d10_gate.shadow_error_count() == rows[0]["shadow_errors_total"]
    kept_live = {r["finding_id"] for r in rows if r["cos_keep"]}
    assert kept_live == {ids[i] for i in ON_TOPIC}
    assert all(r["cos_in_context"] == r["cos_keep"] for r in rows)   # 3 kept < 12, all in the prompt
    assert all(0.0 <= r["mlp_score"] <= 1.0 and r["mlp_keep"] == (r["mlp_score"] >= r["mlp_tau"]) for r in rows)
    assert all(r["gate_version"].endswith(d10_gate.PINNED_MANIFEST_SHA256[:12]) for r in rows)


def test_the_shadow_log_carries_no_text(memory, monkeypatch):
    _seed("d10-notext")
    monkeypatch.setenv("LOCI_D10_SHADOW", "1")
    _run(monkeypatch, "d10-notext")
    raw = _shadow_log(memory).read_text()
    assert raw
    for text in FINDINGS + [QUESTION]:
        assert text not in raw
    for word in ("payment", "OOM", "invoice", "theme", "Ubuntu", "memory"):
        assert word not in raw
    for r in map(json.loads, raw.splitlines()):
        assert not any(k in r for k in ("text", "question", "claim", "evidence", "query"))
        assert all(isinstance(v, (int, float, bool, type(None))) or k in {"ts", "call_id", "investigation_id",
                   "finding_id", "event", "gate_version"} for k, v in r.items())


@pytest.mark.parametrize("fault", ["load", "score", "append", "record"])
def test_shadow_errors_are_swallowed_counted_and_leave_the_decision_alone(memory, monkeypatch, fault):
    _seed("d10-fault")
    monkeypatch.delenv("LOCI_D10_SHADOW", raising=False)
    off = _live_decision(*_run(monkeypatch, "d10-fault"))
    monkeypatch.setenv("LOCI_D10_SHADOW", "1")

    def _raise(*args, **kwargs):
        raise RuntimeError(f"injected {fault} failure")

    if fault == "load":
        monkeypatch.setattr(d10_gate, "load_gate", _raise)
    elif fault == "score":
        monkeypatch.setattr(d10_gate, "pair_features", _raise)
    elif fault == "append":
        import instrumentation_log
        monkeypatch.setattr(instrumentation_log, "append_rows", _raise)
    else:
        monkeypatch.setattr(d10_gate, "record_shadow", _raise)
    before = d10_gate.shadow_error_count()
    on = _live_decision(*_run(monkeypatch, "d10-fault"))
    assert on == off
    assert not _shadow_log(memory).exists()
    if fault != "record":   # a failure outside record_shadow is caught by the server wrapper
        assert d10_gate.shadow_error_count() == before + 1


def test_a_tampered_artifact_in_production_fails_open(memory, monkeypatch, tmp_path):
    _seed("d10-tamper")
    monkeypatch.delenv("LOCI_D10_SHADOW", raising=False)
    off = _live_decision(*_run(monkeypatch, "d10-tamper"))
    d = _copy_artifact(tmp_path)
    (d / d10_gate.MANIFEST_NAME).write_text("{}")
    real = d10_gate.load_gate
    monkeypatch.setattr(d10_gate, "load_gate", lambda: real(d))
    monkeypatch.setenv("LOCI_D10_SHADOW", "1")
    before = d10_gate.shadow_error_count()
    assert _live_decision(*_run(monkeypatch, "d10-tamper")) == off
    assert d10_gate.shadow_error_count() == before + 1
    assert not _shadow_log(memory).exists()


def test_max_pairs_caps_the_logged_rows_by_cosine_rank(memory, monkeypatch):
    _seed("d10-cap")
    monkeypatch.setenv("LOCI_D10_SHADOW", "1")
    monkeypatch.setenv("LOCI_D10_SHADOW_MAX_PAIRS", "2")
    _run(monkeypatch, "d10-cap")
    rows = [json.loads(line) for line in _shadow_log(memory).read_text().splitlines()]
    assert [r["cos_rank"] for r in rows] == [1, 2]
    assert all(r["n_pairs"] == len(FINDINGS) and r["n_logged"] == 2 for r in rows)


def test_record_shadow_itself_never_raises(tmp_path, monkeypatch):
    """The contract holds without the server wrapper: bad input returns False and is counted."""
    monkeypatch.setattr(d10_gate, "_cached", None)
    before = d10_gate.shadow_error_count()
    ok = d10_gate.record_shadow(tmp_path / "log.jsonl", investigation_id="i", question="q",
                                finding_ids=["a", "b"], finding_texts=["x"], vectors=[[1.0]],
                                cosines=[0.1], cos_threshold=0.59, context_ids=[])
    assert ok is False
    assert d10_gate.shadow_error_count() == before + 1
    assert not (tmp_path / "log.jsonl").exists()
