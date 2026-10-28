"""Verify loci is importable as a library without MCP server."""
import sys
import types


def test_policy_import():
    from loci import PolicyTier, PolicyBackend, Policy
    p = Policy(title="test", tier=PolicyTier.LIGHT)
    assert p.tier == PolicyTier.LIGHT


def test_policy_config_cache_fields():
    from loci.policy import PolicyConfig
    cfg = PolicyConfig()
    assert cfg.cache_confidence_threshold == 0.85
    assert cfg.cache_ttl_hours == 24.0


def test_drpg_import():
    from loci import TrajectoryStore, evolve_policy
    assert callable(evolve_policy)


def test_cache_import():
    from loci import check_memory_cache
    result = check_memory_cache([], confidence_threshold=0.9)
    assert result is None


def test_cache_hit_returns_best_score():
    from loci.cache import check_memory_cache, CacheHit
    findings = [
        {"id": "f1", "score": 0.9, "content": "Finding A", "source": "loci_memory"},
        {"id": "f2", "score": 0.95, "content": "Finding B", "source": "loci_memory"},
    ]
    hit = check_memory_cache(findings, confidence_threshold=0.85)
    assert hit is not None
    assert hit.finding_id == "f2"
    assert hit.score == 0.95


def test_graph_cache_import():
    from loci import GraphCache, make_cache_key
    key = make_cache_key("test_tool", {"param": "value"})
    assert len(key) == 16


def test_litellm_import():
    from loci import call_litellm_proxy, LiteLLMResponse
    assert callable(call_litellm_proxy)


def test_litellm_tier_resolve_no_rewrite(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "model_pool", None)   # no pool: the policy map alone decides
    monkeypatch.delenv("LOCI_TIER_MODEL_MAP", raising=False)
    from loci import downgrade_model_to_tier
    from loci.policy import PolicyTier
    model, rewritten = downgrade_model_to_tier("claude-haiku-4.5", PolicyTier.LIGHT)
    assert model == "claude-haiku-4.5"
    assert rewritten is False


def test_litellm_tier_downgrade(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "model_pool", None)
    monkeypatch.delenv("LOCI_TIER_MODEL_MAP", raising=False)
    from loci import downgrade_model_to_tier
    from loci.policy import PolicyTier
    model, rewritten = downgrade_model_to_tier("claude-opus-4.8", PolicyTier.LIGHT)
    assert rewritten is True
    assert model == "claude-haiku-4.5"


def test_investigation_policy_import():
    from loci import cap_perspectives_to_tier, evaluate_pre_answer_for_refusal, filter_findings_for_verification
    capped = cap_perspectives_to_tier(10, "light")
    assert capped == 1


def test_investigation_medium_tier_cap():
    from loci import cap_perspectives_to_tier
    capped = cap_perspectives_to_tier(10, "medium")
    assert capped == 3


def test_evaluate_pre_answer_refusal_no_claims():
    from loci import evaluate_pre_answer_for_refusal
    result = evaluate_pre_answer_for_refusal({})
    assert result["recommend_refusal"] is True
    assert result["r_ref"] == 1.0


def test_evaluate_pre_answer_no_refusal_when_supported():
    from loci import evaluate_pre_answer_for_refusal
    result = evaluate_pre_answer_for_refusal({"supported_count": 9, "unsupported_count": 1})
    assert result["recommend_refusal"] is False
    assert result["supported_ratio"] == 0.9


def test_filter_findings_skip_high_confidence():
    from loci import filter_findings_for_verification
    findings = [
        {"id": "a", "numeric_confidence": 0.95},
        {"id": "b", "numeric_confidence": 0.5},
    ]
    result = filter_findings_for_verification(findings, skip_threshold=0.85)
    assert result["skip_count"] == 1
    assert result["verify_count"] == 1
    assert result["skipped"][0]["id"] == "a"


def test_memory_route_adapter_import():
    from loci import adapt_route_optimization_to_policy, evaluate_counterfactual_gate
    assert callable(adapt_route_optimization_to_policy)


def test_counterfactual_gate_safe():
    from loci import evaluate_counterfactual_gate
    import json
    sim = json.dumps({"changed": 0, "results": [{"id": "r1"}, {"id": "r2"}]})
    result = evaluate_counterfactual_gate(sim, max_changed_fraction=0.05)
    assert result["safe_to_downgrade"] is True
    assert result["changed_fraction"] == 0.0


def test_counterfactual_gate_unsafe():
    from loci import evaluate_counterfactual_gate
    import json
    sim = json.dumps({"changed": 2, "results": [{"id": "r1"}, {"id": "r2"}]})
    result = evaluate_counterfactual_gate(sim, max_changed_fraction=0.05)
    assert result["safe_to_downgrade"] is False


def test_no_server_import_needed():
    """Importing loci must NOT import mcp.server or start any server."""
    import loci  # noqa: F401
    assert "mcp.server" not in sys.modules
