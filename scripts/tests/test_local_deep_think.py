from __future__ import annotations

import logging
import pathlib
import sys


REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import local_deep_think as L  # noqa: E402


def _config(*, red_team: bool = False, self_reflect: bool = True, strict_grounding: bool = False) -> L.ChainConfig:
    return L.ChainConfig(
        topic="fail-open local reasoning",
        investigation_id="local-deep-think-test",
        title="Local deep think test",
        collections=["loci_memory"],
        ideate_models=["fast-model:latest", "strong-model:latest"],
        verify_model="verify-model:latest",
        synthesize_model="synth-model:latest",
        self_reflect_model="reflect-model:latest",
        redteam_model="heretic-model:latest",
        self_reflect=self_reflect,
        red_team=red_team,
        strict_grounding=strict_grounding,
        ideas_per_model=1,
        retrieval_limit=3,
        ground_threshold=0.59,
        evidence_chars=2000,
        max_lineage=6,
    )


def _search_collection(*, query, collection_name, limit):  # noqa: ARG001
    return [
        {"id": "seed-1", "origin": collection_name, "score": 0.91, "text": "Loci prefers fail-open returns."},
        {"id": "seed-2", "origin": collection_name, "score": 0.87, "text": "verify_finding is adversarial."},
    ]


def _gate(query, candidates, threshold):  # noqa: ARG001
    return {"kept": candidates, "dropped": [], "mode": f"cosine>={threshold}"}


def _gate_empty(query, candidates, threshold):  # noqa: ARG001
    return {"kept": [], "dropped": candidates, "mode": f"cosine>={threshold}"}


def _start(**kwargs):  # noqa: ARG001
    return '{"status":"created","manifest":{"id":"local-deep-think-test"}}'


def _load(**kwargs):  # noqa: ARG001
    return '{"findings":[{"id":"f1"},{"id":"f2"}]}'


def test_normal_chain_completion():
    store_calls = []

    def _generate(prompt, *, model="", fmt=None, max_tokens=256, temperature=0.2):  # noqa: ARG001
        if "IDEATE tier" in prompt:
            return {
                "ok": True,
                "text": '{"ideas":[{"claim":"Use dedicated writer persistence.",'
                        '"rationale":"It avoids fabricated store confirmations.",'
                        '"confidence":"high","evidence_ids":["seed-1"]}]}',
            }
        if "SYNTHESIZE tier" in prompt:
            return {
                "ok": True,
                "text": '{"summary":"Best path is dedicated writer plus verify [vf1|verify|verify-model:latest].",'
                        '"supporting_finding_ids":["vf1"],'
                        '"key_findings":[{"finding_id":"vf1","why":"Verified and grounded."}],'
                        '"risks":["Retrieval can still degrade."],'
                        '"next_steps":["Run the red-team tier when needed."]}',
            }
        if "CRITIQUE step" in prompt:
            assert model == "reflect-model:latest"
            return {
                "ok": True,
                "text": "- [vf1] The summary omits that dedicated writer persistence prevents fabricated confirmations.\n"
                        "- [vf1] It should keep the verify gate explicit.",
            }
        if "REVISE step" in prompt:
            return {
                "ok": True,
                "text": '{"summary":"Best path is dedicated writer persistence plus the verify gate '
                        '[vf1|verify|verify-model:latest].",'
                        '"supporting_finding_ids":["vf1"],'
                        '"key_findings":[{"finding_id":"vf1","why":"Verified and grounded."}],'
                        '"risks":["Retrieval can still degrade."],'
                        '"next_steps":["Run the red-team tier when needed."]}',
            }
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")

    def _verify(claim, context="", gen_fn=None, investigation_id=None):  # noqa: ARG001
        assert "fail-open returns" in context
        return {"verdict": "confirmed", "refutation": "could not refute", "confidence": 0.8, "degraded": False}

    def _store(**kwargs):
        store_calls.append(kwargs)
        fid = {1: "f1", 2: "f2", 3: "vf1", 4: "vf2", 5: "sf1"}[len(store_calls)]
        return f'{{"stored": true, "finding_id": "{fid}"}}'

    result = L.run_chain(_config(), deps={
        "generate": _generate,
        "search_collection": _search_collection,
        "gate": _gate,
        "start": _start,
        "load": _load,
        "store": _store,
        "verify": _verify,
    })

    assert result["ideate"]["stored_count"] == 2
    assert result["verify"]["survivor_count"] == 2
    assert result["synthesis"]["finding_id"] == "sf1"
    assert result["synthesis"]["self_reflection"]["revised"] is True
    assert "dedicated writer persistence" in result["synthesis"]["summary"]
    assert store_calls[0]["source"].endswith("#ideate/fast-model:latest")
    assert store_calls[4]["source"].endswith("#self-reflect/reflect-model:latest")
    assert store_calls[2]["derived_from"] == ["f1"]


def test_dead_tier_fails_open():
    def _generate(prompt, *, model="", fmt=None, max_tokens=256, temperature=0.2):  # noqa: ARG001
        if model == "strong-model:latest" and "IDEATE tier" in prompt:
            return {"ok": False, "text": "", "why": "connection refused"}
        if "IDEATE tier" in prompt:
            return {
                "ok": True,
                "text": '{"ideas":[{"claim":"Fallback to remaining tier.",'
                        '"rationale":"One model still answered.","confidence":"medium","evidence_ids":["seed-1"]}]}',
            }
        if "SYNTHESIZE tier" in prompt:
            return {"ok": True, "text": '{"summary":"One tier survived.","supporting_finding_ids":["vf1"]}'}
        return {"ok": False, "text": "", "why": "unexpected"}

    def _verify(claim, context="", gen_fn=None, investigation_id=None):  # noqa: ARG001
        return {"verdict": "confirmed", "refutation": "", "confidence": 0.7, "degraded": False}

    def _store(**kwargs):  # noqa: ARG001
        counter = getattr(_store, "counter", 0) + 1
        _store.counter = counter
        fid = {1: "f1", 2: "vf1", 3: "sf1"}[counter]
        return f'{{"stored": true, "finding_id": "{fid}"}}'

    result = L.run_chain(_config(self_reflect=False), deps={
        "generate": _generate,
        "search_collection": _search_collection,
        "gate": _gate,
        "start": _start,
        "load": _load,
        "store": _store,
        "verify": _verify,
    })

    reports = result["ideate"]["reports"]
    assert any(not report["ok"] for report in reports)
    assert result["ideate"]["stored_count"] == 1
    assert result["verify"]["survivor_count"] == 1


def test_verified_high_confidence_finding_calls_procedure_learning():
    learn_calls = []
    config = _config(self_reflect=False)
    config.ideate_models = ["fast-model:latest"]

    def _generate(prompt, *, model="", fmt=None, max_tokens=256, temperature=0.2):  # noqa: ARG001
        if "IDEATE tier" in prompt:
            return {
                "ok": True,
                "text": '{"ideas":[{"claim":"Restart the worker to clear the stuck queue.",'
                        '"rationale":"It restores progress.","confidence":"high","evidence_ids":["seed-1"]}]}',
            }
        if "SYNTHESIZE tier" in prompt:
            return {"ok": True, "text": '{"summary":"Use the verified restart path.","supporting_finding_ids":["vf1"]}'}
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")

    def _verify(claim, context="", gen_fn=None, investigation_id=None):  # noqa: ARG001
        return {"verdict": "confirmed", "refutation": "", "confidence": 0.91, "degraded": False}

    def _learn_procedure(*, investigation_id, finding_id, verify_verdict, gen_fn=None):
        learn_calls.append({
            "investigation_id": investigation_id,
            "finding_id": finding_id,
            "verify_verdict": verify_verdict,
            "gen_fn": gen_fn,
        })
        return {"promoted": True, "reason": "promoted", "degraded": False}

    def _store(**kwargs):  # noqa: ARG001
        counter = getattr(_store, "counter", 0) + 1
        _store.counter = counter
        fid = {1: "f1", 2: "vf1", 3: "sf1"}[counter]
        return f'{{"stored": true, "finding_id": "{fid}"}}'

    result = L.run_chain(config, deps={
        "generate": _generate,
        "search_collection": _search_collection,
        "gate": _gate,
        "start": _start,
        "load": _load,
        "store": _store,
        "verify": _verify,
        "learn_procedure": _learn_procedure,
    })

    assert len(learn_calls) == 1
    assert learn_calls[0]["investigation_id"] == config.investigation_id
    assert learn_calls[0]["finding_id"] == "f1"
    assert learn_calls[0]["verify_verdict"] == {
        "verdict": "confirmed",
        "refutation": "",
        "confidence": 0.91,
        "degraded": False,
    }
    assert callable(learn_calls[0]["gen_fn"])
    report = result["verify"]["reports"][0]["procedure_learning"]
    assert report["attempted"] is True
    assert report["promoted"] is True


def test_low_confidence_or_unverified_findings_skip_procedure_learning():
    config = _config(self_reflect=False)
    config.ideate_models = ["fast-model:latest"]
    learn_calls = []

    def _generate(prompt, *, model="", fmt=None, max_tokens=256, temperature=0.2):  # noqa: ARG001
        if "IDEATE tier" in prompt:
            return {
                "ok": True,
                "text": '{"ideas":[{"claim":"Restart the worker to clear the stuck queue.",'
                        '"rationale":"It restores progress.","confidence":"high","evidence_ids":["seed-1"]}]}',
            }
        if "SYNTHESIZE tier" in prompt:
            return {"ok": True, "text": '{"summary":"Synth","supporting_finding_ids":["vf1"]}'}
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")

    def _store(**kwargs):  # noqa: ARG001
        counter = getattr(_store, "counter", 0) + 1
        _store.counter = counter
        fid = {1: "f1", 2: "vf1", 3: "sf1"}[counter]
        return f'{{"stored": true, "finding_id": "{fid}"}}'

    def _learn_procedure(*, investigation_id, finding_id, verify_verdict, gen_fn=None):  # noqa: ARG001
        learn_calls.append((investigation_id, finding_id, verify_verdict, gen_fn))
        return {"promoted": True, "reason": "promoted", "degraded": False}

    def _verify_low_confidence(claim, context="", gen_fn=None, investigation_id=None):  # noqa: ARG001
        return {"verdict": "confirmed", "refutation": "", "confidence": 0.41, "degraded": False}

    low_confidence = L.run_chain(config, deps={
        "generate": _generate,
        "search_collection": _search_collection,
        "gate": _gate,
        "start": _start,
        "load": _load,
        "store": _store,
        "verify": _verify_low_confidence,
        "learn_procedure": _learn_procedure,
    })
    assert not learn_calls
    assert low_confidence["verify"]["reports"][0]["procedure_learning"]["reason"] == "confidence_below_threshold"

    _store.counter = 0

    def _verify_unrefuted(claim, context="", gen_fn=None, investigation_id=None):  # noqa: ARG001
        return {"verdict": "refuted", "refutation": "counterexample", "confidence": 0.97, "degraded": False}

    unverified = L.run_chain(config, deps={
        "generate": _generate,
        "search_collection": _search_collection,
        "gate": _gate,
        "start": _start,
        "load": _load,
        "store": _store,
        "verify": _verify_unrefuted,
        "learn_procedure": _learn_procedure,
    })
    assert not learn_calls
    assert unverified["verify"]["reports"][0]["procedure_learning"]["reason"] == "verdict_not_confirmed"
    assert unverified["verify"]["survivor_count"] == 0


def test_procedure_learning_errors_fail_open(caplog):
    config = _config(self_reflect=False)
    config.ideate_models = ["fast-model:latest"]

    def _generate(prompt, *, model="", fmt=None, max_tokens=256, temperature=0.2):  # noqa: ARG001
        if "IDEATE tier" in prompt:
            return {
                "ok": True,
                "text": '{"ideas":[{"claim":"Restart the worker to clear the stuck queue.",'
                        '"rationale":"It restores progress.","confidence":"high","evidence_ids":["seed-1"]}]}',
            }
        if "SYNTHESIZE tier" in prompt:
            return {"ok": True, "text": '{"summary":"Synth","supporting_finding_ids":["vf1"]}'}
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")

    def _verify(claim, context="", gen_fn=None, investigation_id=None):  # noqa: ARG001
        return {"verdict": "confirmed", "refutation": "", "confidence": 0.88, "degraded": False}

    def _learn_procedure(*, investigation_id, finding_id, verify_verdict, gen_fn=None):  # noqa: ARG001
        raise RuntimeError("procedure backend down")

    def _store(**kwargs):  # noqa: ARG001
        counter = getattr(_store, "counter", 0) + 1
        _store.counter = counter
        fid = {1: "f1", 2: "vf1", 3: "sf1"}[counter]
        return f'{{"stored": true, "finding_id": "{fid}"}}'

    caplog.set_level(logging.WARNING)
    result = L.run_chain(config, deps={
        "generate": _generate,
        "search_collection": _search_collection,
        "gate": _gate,
        "start": _start,
        "load": _load,
        "store": _store,
        "verify": _verify,
        "learn_procedure": _learn_procedure,
    })

    assert result["synthesis"]["finding_id"] == "sf1"
    report = result["verify"]["reports"][0]["procedure_learning"]
    assert report["attempted"] is True
    assert report["promoted"] is False
    assert report["reason"] == "unexpected_error"
    assert report["degraded"] is True
    assert "procedure learning failed open for f1" in caplog.text


def test_self_reflection_fail_open_returns_original_synthesis():
    prompts = []

    def _generate(prompt, *, model="", fmt=None, max_tokens=256, temperature=0.2):  # noqa: ARG001
        prompts.append(prompt)
        if "IDEATE tier" in prompt:
            return {"ok": True, "text": '{"ideas":[{"claim":"Idea","rationale":"why","confidence":"medium","evidence_ids":["seed-1"]}]}'}
        if "SYNTHESIZE tier" in prompt:
            return {
                "ok": True,
                "text": '{"summary":"Original synthesis [vf1|verify|verify-model:latest].",'
                        '"supporting_finding_ids":["vf1"]}',
            }
        if "CRITIQUE step" in prompt:
            return {"ok": False, "text": "", "why": "connection refused"}
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")

    def _verify(claim, context="", gen_fn=None, investigation_id=None):  # noqa: ARG001
        return {"verdict": "confirmed", "refutation": "", "confidence": 0.7, "degraded": False}

    def _store(**kwargs):  # noqa: ARG001
        counter = getattr(_store, "counter", 0) + 1
        _store.counter = counter
        fid = {1: "f1", 2: "f2", 3: "vf1", 4: "vf2", 5: "sf1"}[counter]
        return f'{{"stored": true, "finding_id": "{fid}"}}'

    result = L.run_chain(_config(), deps={
        "generate": _generate,
        "search_collection": _search_collection,
        "gate": _gate,
        "start": _start,
        "load": _load,
        "store": _store,
        "verify": _verify,
    })

    assert result["synthesis"]["summary"] == "Original synthesis [vf1|verify|verify-model:latest]."
    assert result["synthesis"]["self_reflection"]["revised"] is False
    assert result["synthesis"]["self_reflection"]["error"] == "connection refused"
    assert sum("CRITIQUE step" in prompt for prompt in prompts) == 1
    assert not any("REVISE step" in prompt for prompt in prompts)


def test_self_reflection_runs_exactly_once():
    prompts = []

    def _generate(prompt, *, model="", fmt=None, max_tokens=256, temperature=0.2):  # noqa: ARG001
        prompts.append(prompt)
        if "IDEATE tier" in prompt:
            return {"ok": True, "text": '{"ideas":[{"claim":"Idea","rationale":"why","confidence":"medium","evidence_ids":["seed-1"]}]}'}
        if "SYNTHESIZE tier" in prompt:
            return {"ok": True, "text": '{"summary":"Synth","supporting_finding_ids":["vf1"]}'}
        if "CRITIQUE step" in prompt:
            return {"ok": True, "text": "- [vf1] Tighten the answer."}
        if "REVISE step" in prompt:
            return {"ok": True, "text": '{"summary":"Revised synth","supporting_finding_ids":["vf1"]}'}
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")

    def _verify(claim, context="", gen_fn=None, investigation_id=None):  # noqa: ARG001
        return {"verdict": "confirmed", "refutation": "", "confidence": 0.7, "degraded": False}

    def _store(**kwargs):  # noqa: ARG001
        counter = getattr(_store, "counter", 0) + 1
        _store.counter = counter
        fid = {1: "f1", 2: "f2", 3: "vf1", 4: "vf2", 5: "sf1"}[counter]
        return f'{{"stored": true, "finding_id": "{fid}"}}'

    result = L.run_chain(_config(), deps={
        "generate": _generate,
        "search_collection": _search_collection,
        "gate": _gate,
        "start": _start,
        "load": _load,
        "store": _store,
        "verify": _verify,
    })

    assert result["synthesis"]["summary"] == "Revised synth"
    assert result["synthesis"]["self_reflection"]["iterations"] == 1
    assert sum("CRITIQUE step" in prompt for prompt in prompts) == 1
    assert sum("REVISE step" in prompt for prompt in prompts) == 1


def test_red_team_flag_controls_critique_tier():
    prompts = []

    def _generate(prompt, *, model="", fmt=None, max_tokens=256, temperature=0.2):  # noqa: ARG001
        prompts.append((model, prompt))
        if "IDEATE tier" in prompt:
            return {"ok": True, "text": '{"ideas":[{"claim":"Idea","rationale":"why","confidence":"medium","evidence_ids":["seed-1"]}]}'}
        if "RED-TEAM tier" in prompt:
            return {"ok": True, "text": '{"critiques":[{"target_finding_id":"vf1","attack":"Exploit weak precondition checks.","severity":"high"}]}'}
        if "SYNTHESIZE tier" in prompt:
            return {"ok": True, "text": '{"summary":"Synth","supporting_finding_ids":["vf1"]}'}
        return {"ok": False, "text": "", "why": "unexpected"}

    def _verify(claim, context="", gen_fn=None, investigation_id=None):  # noqa: ARG001
        return {"verdict": "confirmed", "refutation": "", "confidence": 0.9, "degraded": False}

    def _store(**kwargs):  # noqa: ARG001
        counter = getattr(_store, "counter", 0) + 1
        _store.counter = counter
        fid = {1: "f1", 2: "f2", 3: "vf1", 4: "rf1", 5: "sf1"}[counter]
        return f'{{"stored": true, "finding_id": "{fid}"}}'

    off = L.run_chain(_config(red_team=False, self_reflect=False), deps={
        "generate": _generate,
        "search_collection": _search_collection,
        "gate": _gate,
        "start": _start,
        "load": _load,
        "store": _store,
        "verify": _verify,
    })
    assert off["red_team"]["enabled"] is False
    assert not any("RED-TEAM tier" in prompt for _, prompt in prompts)

    prompts.clear()
    _store.counter = 0
    on = L.run_chain(_config(red_team=True, self_reflect=False), deps={
        "generate": _generate,
        "search_collection": _search_collection,
        "gate": _gate,
        "start": _start,
        "load": _load,
        "store": _store,
        "verify": _verify,
    })
    assert on["red_team"]["enabled"] is True
    assert on["red_team"]["stored"] == 1
    assert any(model == "heretic-model:latest" and "RED-TEAM tier" in prompt for model, prompt in prompts)


def test_dedicated_writer_owns_persistence_calls():
    stores = []

    def _generate(prompt, *, model="", fmt=None, max_tokens=256, temperature=0.2):  # noqa: ARG001
        if "IDEATE tier" in prompt:
            return {"ok": True, "text": '{"ideas":[{"claim":"Writer owns stores","rationale":"workflow evidence","confidence":"high","evidence_ids":["seed-1"]}]}'}
        if "SYNTHESIZE tier" in prompt:
            return {"ok": True, "text": '{"summary":"done","supporting_finding_ids":["vf1"]}'}
        return {"ok": False, "text": "", "why": "unexpected"}

    def _verify(claim, context="", gen_fn=None, investigation_id=None):  # noqa: ARG001
        return {"verdict": "confirmed", "refutation": "", "confidence": 0.8, "degraded": False}

    def _store(**kwargs):
        stores.append(kwargs)
        counter = len(stores)
        fid = {1: "f1", 2: "f2", 3: "vf1"}[counter]
        return f'{{"stored": true, "finding_id": "{fid}"}}'

    L.run_chain(_config(self_reflect=False), deps={
        "generate": _generate,
        "search_collection": _search_collection,
        "gate": _gate,
        "start": _start,
        "load": _load,
        "store": _store,
        "verify": _verify,
    })

    assert all(call["source"].startswith("scripts/local_deep_think.py#") for call in stores)
    assert stores[0]["derived_from"] is None
    assert stores[2]["derived_from"] == ["f1"]
    assert stores[0]["metadata"]["evidence_provenance_tier"] == "model_asserted"
    assert stores[2]["metadata"]["support_evidence_refs"] == ["seed-1", "seed-2"]


def test_verify_findings_blocks_model_only_provenance_support():
    verify_calls = []

    def _search(*, query, collection_name, limit):  # noqa: ARG001
        return [{
            "id": "model-memory-1",
            "origin": collection_name,
            "score": 0.95,
            "text": "Circular claim from a prior model run.",
            "metadata": {"evidence_provenance_tier": "model_asserted"},
        }]

    def _verify(claim, context="", gen_fn=None, investigation_id=None):  # noqa: ARG001
        verify_calls.append(claim)
        return {"verdict": "confirmed", "refutation": "", "confidence": 0.9, "degraded": False}

    survivors, reports = L.verify_findings(
        [L.StoredFinding("idea-1", "Circular claim from a prior model run.", "ideate", "m")],
        topic="circularity",
        config=_config(self_reflect=False),
        search_fn=_search,
        gate_fn=_gate,
        verify_fn=_verify,
        gen_fn=lambda *args, **kwargs: {"ok": True, "text": "{}"},
        store_fn=lambda **kwargs: '{"stored": true, "finding_id": "vf1"}',
    )

    assert survivors == []
    assert verify_calls == []
    assert reports[0]["verdict"] == "uncertain"
    assert reports[0]["provenance_firewall"]["allowed"] is False


def _ungrounded_deps():
    """generate/verify/store stack shared by the strict-grounding tests below.

    Uses _gate_empty so every retrieval (top-level and per-claim verify) reports
    kept=0 -- i.e. nothing in this run ever grounded against real evidence.
    """
    stores = []

    def _generate(prompt, *, model="", fmt=None, max_tokens=256, temperature=0.2):  # noqa: ARG001
        if "IDEATE tier" in prompt:
            return {
                "ok": True,
                "text": '{"ideas":[{"claim":"Idea with no supporting evidence.",'
                        '"rationale":"speculative","confidence":"medium","evidence_ids":[]}]}',
            }
        if "SYNTHESIZE tier" in prompt:
            return {
                "ok": True,
                "text": '{"summary":"Confident-sounding synthesis.","supporting_finding_ids":["vf1"]}',
            }
        return {"ok": False, "text": "", "why": "unexpected"}

    def _verify(claim, context="", gen_fn=None, investigation_id=None):  # noqa: ARG001
        return {"verdict": "confirmed", "refutation": "", "confidence": 0.6, "degraded": False}

    def _store(**kwargs):
        stores.append(kwargs)
        counter = len(stores)
        fid = {1: "f1", 2: "vf1", 3: "sf1"}[counter]
        return f'{{"stored": true, "finding_id": "{fid}"}}'

    deps = {
        "generate": _generate,
        "search_collection": _search_collection,
        "gate": _gate_empty,
        "start": _start,
        "load": _load,
        "store": _store,
        "verify": _verify,
    }
    return deps, stores


def test_default_mode_unaffected_by_ungrounded_run():
    deps, stores = _ungrounded_deps()
    result = L.run_chain(_config(self_reflect=False, strict_grounding=False), deps=deps)

    synthesis = result["synthesis"]
    assert synthesis["grounding_status"] == "ungrounded"
    assert synthesis["summary"] == "Confident-sounding synthesis."
    assert "ungrounded" not in (synthesis.get("tags") or [])
    synth_store = stores[-1]
    assert synth_store["confidence"] == "high"


def test_strict_grounding_downgrades_ungrounded_synthesis():
    deps, stores = _ungrounded_deps()
    result = L.run_chain(_config(self_reflect=False, strict_grounding=True), deps=deps)

    synthesis = result["synthesis"]
    assert synthesis["grounding_status"] == "ungrounded"
    assert "ungrounded" in synthesis["tags"]
    assert synthesis["summary"].startswith(L._UNGROUNDED_CAVEAT)
    assert "Confident-sounding synthesis." in synthesis["summary"]
    synth_store = stores[-1]
    assert synth_store["confidence"] == "low"


def test_strict_grounding_leaves_grounded_synthesis_untouched():
    def _generate(prompt, *, model="", fmt=None, max_tokens=256, temperature=0.2):  # noqa: ARG001
        if "IDEATE tier" in prompt:
            return {
                "ok": True,
                "text": '{"ideas":[{"claim":"Idea with real evidence.",'
                        '"rationale":"grounded","confidence":"high","evidence_ids":["seed-1"]}]}',
            }
        if "SYNTHESIZE tier" in prompt:
            return {"ok": True, "text": '{"summary":"Grounded synthesis.","supporting_finding_ids":["vf1"]}'}
        return {"ok": False, "text": "", "why": "unexpected"}

    def _verify(claim, context="", gen_fn=None, investigation_id=None):  # noqa: ARG001
        return {"verdict": "confirmed", "refutation": "", "confidence": 0.8, "degraded": False}

    stores = []

    def _store(**kwargs):
        stores.append(kwargs)
        counter = len(stores)
        fid = {1: "f1", 2: "vf1", 3: "sf1"}[counter]
        return f'{{"stored": true, "finding_id": "{fid}"}}'

    result = L.run_chain(_config(self_reflect=False, strict_grounding=True), deps={
        "generate": _generate,
        "search_collection": _search_collection,
        "gate": _gate,
        "start": _start,
        "load": _load,
        "store": _store,
        "verify": _verify,
    })

    synthesis = result["synthesis"]
    assert synthesis["grounding_status"] == "grounded"
    assert synthesis["summary"] == "Grounded synthesis."
    assert "ungrounded" not in synthesis["tags"]
    assert stores[-1]["confidence"] == "high"
