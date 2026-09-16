from __future__ import annotations

import pathlib
import sys


REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import local_deep_think as L  # noqa: E402


def _config(*, red_team: bool = False) -> L.ChainConfig:
    return L.ChainConfig(
        topic="fail-open local reasoning",
        investigation_id="local-deep-think-test",
        title="Local deep think test",
        collections=["loci_memory"],
        ideate_models=["fast-model:latest", "strong-model:latest"],
        verify_model="verify-model:latest",
        synthesize_model="synth-model:latest",
        redteam_model="heretic-model:latest",
        red_team=red_team,
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
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")

    def _verify(claim, context="", gen_fn=None, investigation_id=None):  # noqa: ARG001
        assert "fail-open returns" in context
        return {"verdict": "confirmed", "refutation": "could not refute", "confidence": 0.8, "degraded": False}

    def _store(**kwargs):
        store_calls.append(kwargs)
        fid = f"f{len(store_calls)}" if len(store_calls) <= 2 else ("vf1" if len(store_calls) == 3 else "sf1")
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
    assert store_calls[0]["source"].endswith("#ideate/fast-model:latest")
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

    result = L.run_chain(_config(), deps={
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

    off = L.run_chain(_config(red_team=False), deps={
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
    on = L.run_chain(_config(red_team=True), deps={
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

    L.run_chain(_config(), deps={
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
