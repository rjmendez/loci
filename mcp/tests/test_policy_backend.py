"""
Test-driven implementation for Loci policy.md backend.

Tests cover:
1. Policy loading/storing via LociMCP tools
2. Tier-aware policy tracking
3. Environment defaults (LOCI_POLICY_* variables)
4. Hook invocation (loci_policy_hook, grounding_hook augmentation)
5. Policy validation against tier/region constraints
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add mcp directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loci.hooks import (
    eval_hook_stub,
    grounding_hook_augmented,
    loci_policy_hook,
    tier_router_surface,
)
from loci.policy import (
    Policy,
    PolicyBackend,
    PolicyRegion,
    PolicyTier,
    get_policy_config,
    validate_policy,
)


class TestPolicyLoading(unittest.TestCase):
    """GREEN: Policy loading from policy.md files and storage."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.policy_dir = Path(self.tmp_dir.name) / "policies"
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        self.backend = PolicyBackend(self.policy_dir)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_policy_store_creates_policy_json_with_metadata(self):
        """Test: policy_store creates a policy.json artifact with type, labels, tier, region."""
        policy = Policy(
            title="EVO-RAG Discovery",
            tier=PolicyTier.LIGHT,
            region=PolicyRegion.MEMORY,
            labels=["drpg", "evo-rag"],
            config={"discovery_mode": True}
        )
        
        result = self.backend.store(policy)
        
        self.assertEqual(result["status"], "stored")
        self.assertEqual(result["policy_id"], policy.policy_id)
        
        # Verify file was created
        policy_file = self.policy_dir / f"{policy.policy_id}.json"
        self.assertTrue(policy_file.exists())
        
        # Verify metadata
        with open(policy_file) as f:
            stored = json.load(f)
        
        self.assertEqual(stored["type"], "policy.md")
        self.assertEqual(stored["tier"], "light")
        self.assertEqual(stored["region"], "memory")
        self.assertIn("metadata", stored)
        self.assertEqual(stored["metadata"]["version"], 1)

    def test_policy_load_retrieves_policy_by_id(self):
        """Test: policy_load retrieves a stored policy and deserializes it."""
        original = Policy(
            title="Test Policy",
            tier=PolicyTier.MEDIUM,
            region=PolicyRegion.CONTEXT,
            labels=["test"],
            config={"key": "value"}
        )
        
        self.backend.store(original)
        loaded = self.backend.load(original.policy_id)
        
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.policy_id, original.policy_id)
        self.assertEqual(loaded.title, original.title)
        self.assertEqual(loaded.tier, original.tier)
        self.assertEqual(loaded.region, original.region)
        self.assertEqual(loaded.labels, original.labels)
        self.assertEqual(loaded.config, original.config)

    def test_policy_load_nonexistent_policy_returns_none(self):
        """Test: policy_load on missing policy_id returns None."""
        loaded = self.backend.load("nonexistent-id")
        self.assertIsNone(loaded)

    def test_policy_store_validates_required_fields(self):
        """Test: policy_store rejects policies missing title."""
        policy = Policy(
            title="",  # Missing title
            tier=PolicyTier.LIGHT,
            region=PolicyRegion.MEMORY
        )
        
        with self.assertRaisesRegex(ValueError, "missing_title"):
            self.backend.store(policy)


class TestEnvironmentDefaults(unittest.TestCase):
    """GREEN: Environment variable defaults for policy configuration."""

    def setUp(self):
        self.original_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_loci_policy_enable_default_false(self):
        """Test: LOCI_POLICY_ENABLE defaults to False when unset."""
        os.environ.pop("LOCI_POLICY_ENABLE", None)
        config = get_policy_config()
        self.assertFalse(config.enabled)

    def test_loci_policy_enable_true_from_env(self):
        """Test: LOCI_POLICY_ENABLE=1 or true enables policy tracking."""
        os.environ["LOCI_POLICY_ENABLE"] = "1"
        config = get_policy_config()
        self.assertTrue(config.enabled)
        
        os.environ["LOCI_POLICY_ENABLE"] = "true"
        config = get_policy_config()
        self.assertTrue(config.enabled)

    def test_loci_policy_max_chars_default_8192(self):
        """Test: LOCI_POLICY_MAX_CHARS defaults to 8192."""
        os.environ.pop("LOCI_POLICY_MAX_CHARS", None)
        config = get_policy_config()
        self.assertEqual(config.max_chars, 8192)

    def test_loci_policy_max_chars_from_env_override(self):
        """Test: LOCI_POLICY_MAX_CHARS=16384 overrides default."""
        os.environ["LOCI_POLICY_MAX_CHARS"] = "16384"
        config = get_policy_config()
        self.assertEqual(config.max_chars, 16384)

    def test_loci_policy_allowed_models_comma_separated(self):
        """Test: LOCI_POLICY_ALLOWED_MODELS=model1,model2 parses correctly."""
        os.environ["LOCI_POLICY_ALLOWED_MODELS"] = "gpt-4,claude-3,llama-70b"
        config = get_policy_config()
        self.assertEqual(config.allowed_models, ["gpt-4", "claude-3", "llama-70b"])

    def test_loci_policy_allowed_models_default_unrestricted(self):
        """Test: LOCI_POLICY_ALLOWED_MODELS unset allows all models."""
        os.environ.pop("LOCI_POLICY_ALLOWED_MODELS", None)
        config = get_policy_config()
        self.assertIsNone(config.allowed_models)


class TestTierAwareTracking(unittest.TestCase):
    """GREEN: Tier-aware policy tracking (light, medium, heavy)."""

    def test_policy_tier_light_constraints(self):
        """Test: light tier policy limits context to 2K chars, 3 hops max."""
        light_policy = Policy(
            title="Light Tier Test",
            tier=PolicyTier.LIGHT,
            config={"max_context_chars": 2048, "max_retrieval_hops": 3}
        )
        
        constraints = light_policy.get_constraints()
        self.assertEqual(constraints["max_context_chars"], 2048)
        self.assertEqual(constraints["max_retrieval_hops"], 3)
        self.assertEqual(constraints["max_refinement_steps"], 2)

    def test_policy_tier_medium_balanced(self):
        """Test: medium tier balances retrieval and context (8K, 7 hops)."""
        medium_policy = Policy(
            title="Medium Tier Test",
            tier=PolicyTier.MEDIUM,
            config={"max_context_chars": 8192, "max_retrieval_hops": 7}
        )
        
        constraints = medium_policy.get_constraints()
        self.assertEqual(constraints["max_context_chars"], 8192)
        self.assertEqual(constraints["max_retrieval_hops"], 7)
        self.assertEqual(constraints["max_refinement_steps"], 4)

    def test_policy_tier_heavy_unrestricted(self):
        """Test: heavy tier allows large context and deep retrieval."""
        heavy_policy = Policy(
            title="Heavy Tier Test",
            tier=PolicyTier.HEAVY,
            config={"max_context_chars": 65536, "max_retrieval_hops": 15}
        )
        
        constraints = heavy_policy.get_constraints()
        self.assertEqual(constraints["max_context_chars"], 65536)
        self.assertEqual(constraints["max_retrieval_hops"], 15)
        self.assertEqual(constraints["max_refinement_steps"], 10)

    def test_policy_region_memory_vs_context_vs_routing(self):
        """Test: policy regions (memory, context, routing) have distinct scopes."""
        memory_policy = Policy(
            title="Memory Region Test",
            region=PolicyRegion.MEMORY,
            config={"discovery_mode": True, "reward_vector_enabled": True}
        )
        
        self.assertEqual(memory_policy.region, PolicyRegion.MEMORY)
        
        context_policy = Policy(region=PolicyRegion.CONTEXT)
        self.assertEqual(context_policy.region, PolicyRegion.CONTEXT)
        
        routing_policy = Policy(region=PolicyRegion.ROUTING)
        self.assertEqual(routing_policy.region, PolicyRegion.ROUTING)

    def test_policy_get_reward_vector_discovery(self):
        """Test: Discovery phase uses high retrieval reward vector weights."""
        discovery_policy = Policy(
            title="Discovery Phase",
            tier=PolicyTier.LIGHT,
            config={"discovery_mode": True}
        )
        
        weights = discovery_policy.get_reward_vector_weights()
        
        self.assertAlmostEqual(weights["r_ret"], 1.0)   # High retrieval incentive
        self.assertAlmostEqual(weights["r_rem"], -0.05)  # Low search penalty
        self.assertAlmostEqual(weights["r_ref"], -0.5)   # Discourage refusal
        self.assertAlmostEqual(weights["r_cor"], 0.5)   # Moderate correctness

    def test_policy_get_reward_vector_refinement(self):
        """Test: Refinement phase prioritizes efficiency."""
        refinement_policy = Policy(
            title="Refinement Phase",
            tier=PolicyTier.MEDIUM,
            config={"discovery_mode": False}
        )
        
        weights = refinement_policy.get_reward_vector_weights()
        
        self.assertAlmostEqual(weights["r_ret"], 0.5)     # Moderate retrieval
        self.assertAlmostEqual(weights["r_rem"], -0.2)    # Higher search penalty
        self.assertAlmostEqual(weights["r_ref"], 1.0)     # High refusal reward
        self.assertAlmostEqual(weights["r_cor"], 1.0)    # High correctness

    def test_policy_tier_standard_balanced(self):
        """Test: STANDARD tier has 8K context, 7 hops, 4 refinement steps."""
        policy = Policy(
            title="Standard Tier Test",
            tier=PolicyTier.STANDARD,
            config={"max_context_chars": 8192, "max_retrieval_hops": 7},
        )
        c = policy.get_constraints()
        self.assertEqual(c["max_context_chars"], 8192)
        self.assertEqual(c["max_retrieval_hops"], 7)
        self.assertEqual(c["max_refinement_steps"], 4)

    def test_policy_tier_medium_equals_standard(self):
        """Test: MEDIUM is a backward-compat alias for STANDARD constraints."""
        policy = Policy(title="Medium Alias Test", tier=PolicyTier.MEDIUM)
        c = policy.get_constraints()
        self.assertEqual(c["max_context_chars"], 8192)
        self.assertEqual(c["max_retrieval_hops"], 7)

    def test_policy_tier_zero_all_zeros(self):
        """Test: ZERO tier returns zero-cost reward vector."""
        policy = Policy(title="Zero Tier", tier=PolicyTier.ZERO, config={"discovery_mode": True})
        weights = policy.get_reward_vector_weights()
        for key, val in weights.items():
            self.assertEqual(val, 0.0, f"{key} should be 0.0 for ZERO tier")

    def test_policy_tier_nano_lightweight_weights(self):
        """Test: NANO tier returns lightweight discovery weights."""
        policy = Policy(title="Nano Tier", tier=PolicyTier.NANO)
        weights = policy.get_reward_vector_weights()
        self.assertAlmostEqual(weights["r_ret"], 0.8)
        self.assertAlmostEqual(weights["r_cor"], 0.4)

    def test_policy_tier_turbo_refinement_weights(self):
        """Test: TURBO tier returns refinement-leaning weights."""
        policy = Policy(title="Turbo Tier", tier=PolicyTier.TURBO)
        weights = policy.get_reward_vector_weights()
        self.assertAlmostEqual(weights["r_cor"], 0.9)
        self.assertAlmostEqual(weights["r_ref"], 0.7)

    def test_policy_tier_research_high_backtrack_penalty(self):
        """Test: RESEARCH tier has high backtrack penalty (expensive operations)."""
        policy = Policy(title="Research Tier", tier=PolicyTier.RESEARCH)
        weights = policy.get_reward_vector_weights()
        self.assertAlmostEqual(weights["r_bkp"], -0.6)
        self.assertAlmostEqual(weights["r_ret"], 1.0)


class TestLociMCPTools(unittest.TestCase):
    """GREEN: LociMCP mcp__loci__policy_store and mcp__loci__policy_load tools."""

    async def async_loci_policy_hook_test(self):
        """Test: loci_policy_hook returns (apply: bool, policy_id: str|None, reason: str)."""
        # Test: disabled env
        os.environ["LOCI_POLICY_ENABLE"] = "0"
        config = get_policy_config()
        self.assertFalse(config.enabled)
        
        # Test: enabled but model not allowed
        os.environ["LOCI_POLICY_ENABLE"] = "1"
        os.environ["LOCI_POLICY_ALLOWED_MODELS"] = "gpt-4"
        result = await loci_policy_hook(
            model="claude-3",
            messages=[{"role": "user", "content": "test"}],
            kwargs={},
            litellm_params={}
        )
        self.assertFalse(result["apply"])
        self.assertEqual(result["reason"], "model_not_allowed")
        os.environ.pop("LOCI_POLICY_ALLOWED_MODELS", None)

    async def async_grounding_hook_test(self):
        """Test: grounding_hook augmented to surface policy's backing evidence."""
        backend = PolicyBackend()
        policy = Policy(
            title="Test Policy",
            tier=PolicyTier.LIGHT,
            region=PolicyRegion.MEMORY,
            config={"discovery_mode": True}
        )
        backend.store(policy)
        
        messages = [{"role": "user", "content": "test"}]
        kwargs = {"policy_id": policy.policy_id, "grounding_context": {"findings": [{"score": 0.9, "text": "matched doc"}]}}
        
        result = await grounding_hook_augmented(
            model="gpt-4",
            messages=messages,
            kwargs=kwargs,
            litellm_params={}
        )
        
        self.assertTrue(result["augmented"])
        self.assertIn("POLICY-AWARE MEMORY MATCH", result["messages"][0]["content"])
        self.assertIn("Tier: light", result["messages"][0]["content"])
        self.assertIn("Reward Vector", result["messages"][0]["content"])

    async def async_tier_router_tests(self):
        """Test: tier_router with a LIGHT-tier model and LIGHT policy — no downgrade."""
        backend = PolicyBackend()
        policy = Policy(
            title="Tier Constraint Test",
            tier=PolicyTier.LIGHT,
            region=PolicyRegion.ROUTING
        )
        backend.store(policy)

        messages = []
        # claude-haiku-4.5 is a LIGHT-tier model; policy is also LIGHT → no rewrite
        kwargs = {"policy_id": policy.policy_id}

        result = await tier_router_surface(
            model="claude-haiku-4.5",
            messages=messages,
            kwargs=kwargs,
            litellm_params={}
        )

        self.assertFalse(result["downgraded"])
        self.assertEqual(result["tier_decision"], "light")
        self.assertEqual(result["model"], "claude-haiku-4.5")

    async def async_tier_router_downgrade_tests(self):
        """Test: tier_router downgrades a HEAVY model against a LIGHT policy."""
        backend = PolicyBackend()
        policy = Policy(
            title="Tier Constraint Test",
            tier=PolicyTier.LIGHT,
            region=PolicyRegion.ROUTING
        )
        backend.store(policy)

        messages = []
        # claude-opus-4.8 is HEAVY; policy is LIGHT → must downgrade
        kwargs = {"policy_id": policy.policy_id}

        result = await tier_router_surface(
            model="claude-opus-4.8",
            messages=messages,
            kwargs=kwargs,
            litellm_params={}
        )

        self.assertTrue(result["downgraded"])
        self.assertEqual(result["tier_decision"], "light")
        self.assertEqual(result["model"], "claude-haiku-4.5")
        self.assertIn("model_downgraded", [v["type"] for v in result["violations"]])

    def test_tier_router_surfaces_policy_constraints(self):
        """Test wrapper for async test."""
        asyncio.run(self.async_tier_router_tests())

    def test_tier_router_downgrades_on_exceed_tier(self):
        """Test wrapper for downgrade test."""
        asyncio.run(self.async_tier_router_downgrade_tests())


class TestHookIntegration(unittest.TestCase):
    """GREEN: LiteLLM hooks integration (policy_hook, grounding_hook augment, tier_router)."""

    def setUp(self):
        self.original_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_loci_policy_hook_decides_whether_to_apply_policy(self):
        """Test: loci_policy_hook returns apply=False when disabled, apply=True when enabled."""
        # Negative: policy disabled (default) → apply=False with reason
        os.environ.pop("LOCI_POLICY_ENABLE", None)
        result = asyncio.run(loci_policy_hook("gpt-4", [], {}, {}))
        self.assertFalse(result["apply"])
        self.assertEqual(result["reason"], "policy_disabled_via_env")

        # Positive: policy enabled → apply=True
        os.environ["LOCI_POLICY_ENABLE"] = "1"
        result = asyncio.run(loci_policy_hook("gpt-4", [], {}, {}))
        self.assertTrue(result["apply"])
        self.assertEqual(result["reason"], "policy_enabled")

    def test_grounding_hook_augmented_with_policy_backing(self):
        """Test: grounding_hook returns messages unchanged when no policy; augments when one is set."""
        messages = [{"role": "user", "content": "query"}]

        # Negative: no policy_id → not augmented, messages pass through unchanged
        result = asyncio.run(grounding_hook_augmented("gpt-4", messages, {}, {}))
        self.assertEqual(result["messages"], messages)
        self.assertFalse(result["augmented"])

        # Positive: policy_id provided with mocked backend → augmented=True
        policy = Policy(
            title="Test Policy",
            tier=PolicyTier.LIGHT,
            region=PolicyRegion.MEMORY,
            config={"discovery_mode": True},
        )
        with patch("loci.hooks.PolicyBackend") as mock_backend_cls:
            mock_backend_cls.return_value.load.return_value = policy
            result = asyncio.run(
                grounding_hook_augmented("gpt-4", messages, {"policy_id": "p-test"}, {})
            )
        self.assertTrue(result["augmented"])
        self.assertEqual(result["policy_id"], "p-test")

    def test_tier_router_surfaces_policy_constraints(self):
        """Test: tier_router returns default when no policy; downgrades when tier exceeds policy."""
        # Negative: no policy_id → no downgrade
        result = asyncio.run(tier_router_surface("gpt-4", [], {}, {}))
        self.assertEqual(result["tier_decision"], "default")
        self.assertFalse(result["downgraded"])

        # Positive: policy tier=light, requested tier=heavy → downgraded to light
        policy = Policy(
            title="Light Policy",
            tier=PolicyTier.LIGHT,
            region=PolicyRegion.ROUTING,
        )
        with patch("loci.hooks.PolicyBackend") as mock_backend_cls:
            mock_backend_cls.return_value.load.return_value = policy
            result = asyncio.run(
                tier_router_surface("gpt-4", [], {"policy_id": "p1", "tier": "heavy"}, {})
            )
        self.assertTrue(result["downgraded"])
        self.assertEqual(result["tier_decision"], "light")

    def test_eval_hook_stub_captures_feedback_for_drpg(self):
        """Test: eval_hook_stub returns logged=True with step_id and full reward vector."""
        reward_vector = {
            "r_ret": 1.0, "r_rem": -0.1, "r_ovp": 0.0,
            "r_bkp": 0.0, "r_ref": 0.0, "r_step": -0.05, "r_cor": 0.0,
        }
        kwargs = {
            "step": 1, "action": "SEARCH",
            "reward_vector": reward_vector, "policy_id": "policy-abc",
        }
        result = asyncio.run(eval_hook_stub("gpt-4", [], kwargs, {}))

        self.assertTrue(result["logged"])
        self.assertEqual(result["step_id"], "policy-abc:step:1")
        feedback = result["feedback"]
        self.assertEqual(feedback["step"], 1)
        self.assertEqual(feedback["action"], "SEARCH")
        self.assertEqual(feedback["reward"]["r_ret"], 1.0)
        self.assertEqual(feedback["reward"]["r_step"], -0.05)


class TestPolicyValidation(unittest.TestCase):
    """GREEN: Policy validation against tier/region constraints."""

    def test_policy_validation_rejects_invalid_tier(self):
        """Test: policy validator rejects tier not in (light, medium, heavy)."""
        # Negative: invalid tier → error with code invalid_tier
        errors = validate_policy({"tier": "extreme", "region": "memory", "title": "x", "config": {}})
        self.assertIn("invalid_tier", [e["code"] for e in errors])

        # Positive: valid tier → no invalid_tier error
        errors = validate_policy({"tier": "light", "region": "memory", "title": "x", "config": {}})
        self.assertNotIn("invalid_tier", [e["code"] for e in errors])

    def test_policy_validation_rejects_invalid_region(self):
        """Test: policy validator rejects region not in (memory, context, routing)."""
        # Negative: invalid region → error with code invalid_region
        errors = validate_policy({"tier": "light", "region": "unknown", "title": "x", "config": {}})
        self.assertIn("invalid_region", [e["code"] for e in errors])

        # Positive: valid region → no invalid_region error
        errors = validate_policy({"tier": "light", "region": "context", "title": "x", "config": {}})
        self.assertNotIn("invalid_region", [e["code"] for e in errors])

    def test_policy_validation_checks_config_consistency(self):
        """Test: max_context_chars above light-tier limit (2048) triggers tier_config_mismatch."""
        # Negative: config exceeds light-tier limit
        errors = validate_policy({"tier": "light", "title": "x", "config": {"max_context_chars": 100000}})
        self.assertTrue(any(e["code"] == "tier_config_mismatch" for e in errors))

        # Positive: config within light-tier limit → no mismatch
        errors = validate_policy({"tier": "light", "title": "x", "config": {"max_context_chars": 1024}})
        self.assertFalse(any(e["code"] == "tier_config_mismatch" for e in errors))

    def test_new_tiers_pass_validation(self):
        """Test: all 7 new tiers (zero, nano, standard, turbo, research) are valid."""
        for tier_str in ("zero", "nano", "standard", "turbo", "research"):
            with self.subTest(tier=tier_str):
                errors = validate_policy({"title": "x", "tier": tier_str, "region": "memory"})
                tier_errors = [e for e in errors if e["code"] == "invalid_tier"]
                self.assertFalse(tier_errors, f"{tier_str} flagged as invalid tier")


class TestDRPGIntegration(unittest.TestCase):
    """GREEN: DRPG discovery↔refinement and 7D reward vector integration."""

    def test_policy_discovery_phase_uses_high_retrieval_bonus(self):
        """Test: Discovery phase policy has r_ret=1.0, r_rem=-0.05, r_bkp=-0.2."""
        policy = Policy(
            title="Discovery Policy",
            tier=PolicyTier.MEDIUM,
            region=PolicyRegion.MEMORY,
            config={"discovery_mode": True},
        )
        weights = policy.get_reward_vector_weights()
        self.assertEqual(weights["r_ret"], 1.0)
        self.assertEqual(weights["r_rem"], -0.05)
        self.assertEqual(weights["r_bkp"], -0.2)

    def test_policy_refinement_phase_prioritizes_efficiency(self):
        """Test: Refinement phase policy has r_ret=0.5, r_rem=-0.2, r_ref=1.0."""
        policy = Policy(
            title="Refinement Policy",
            tier=PolicyTier.MEDIUM,
            region=PolicyRegion.MEMORY,
            config={"discovery_mode": False},
        )
        weights = policy.get_reward_vector_weights()
        self.assertEqual(weights["r_ret"], 0.5)
        self.assertEqual(weights["r_rem"], -0.2)
        self.assertEqual(weights["r_ref"], 1.0)

    def test_reward_vector_step_tracking(self):
        """Test: eval_hook_stub returns all 7 reward components logged for DRPG."""
        reward_vector = {
            "r_ret": 1.0, "r_rem": -0.1, "r_ovp": 0.0,
            "r_bkp": 0.0, "r_ref": 0.0, "r_step": -0.05, "r_cor": 0.8,
        }
        kwargs = {"step": 3, "action": "ANSWER", "reward_vector": reward_vector, "policy_id": "drpg-1"}
        result = asyncio.run(eval_hook_stub("gpt-4", [], kwargs, {}))

        self.assertTrue(result["logged"])
        feedback = result["feedback"]
        self.assertEqual(len(feedback["reward"]), 7)
        self.assertEqual(feedback["reward"]["r_cor"], 0.8)
        self.assertEqual(feedback["action"], "ANSWER")


class TestToolCallEnforcementNoToolsEnv(unittest.TestCase):
    """Fix 2: allowed_models must NOT be reused as fallback tool allowlist.

    Mutations run:
    - Reintroduced the `elif policy_config.allowed_models: allowed_tools = ...`
      branch → tool_not_allowed returned for any tool call, test_passes_all_tools
      failed.
    - Also verified positive twin: with LOCI_POLICY_ALLOWED_TOOLS set, blocking
      still fires (test_blocks_disallowed_tool passes only if the env var path works).
    """

    def _run_hook(self, tool_name, env_overrides):
        async def _inner():
            return await loci_policy_hook(
                model="claude-haiku-4.5",
                messages=[],
                kwargs={"tool_call_name": tool_name},
                litellm_params={},
            )
        with patch.dict(os.environ, env_overrides, clear=False):
            return asyncio.run(_inner())

    def test_passes_all_tools_when_only_allowed_models_set(self):
        """LOCI_POLICY_ALLOWED_TOOLS unset, LOCI_POLICY_ALLOWED_MODELS set → any tool passes."""
        env = {
            "LOCI_POLICY_ENABLE": "1",
            "LOCI_POLICY_ALLOWED_MODELS": "claude-haiku-4.5,gpt-5-mini",
        }
        with patch.dict(os.environ, env, clear=False):
            # Remove LOCI_POLICY_ALLOWED_TOOLS if present
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("LOCI_POLICY_ALLOWED_TOOLS", None)
                result = asyncio.run(loci_policy_hook(
                    model="claude-haiku-4.5",
                    messages=[],
                    kwargs={"tool_call_name": "some_arbitrary_tool_not_in_models_list"},
                    litellm_params={},
                ))
        self.assertEqual(result["reason"], "policy_enabled")
        self.assertNotEqual(result["reason"], "tool_not_allowed")
        self.assertTrue(result["apply"])
        violations = [v for v in result["violations"] if v["type"] == "tool_not_allowed"]
        self.assertEqual(violations, [])

    def test_blocks_disallowed_tool_when_allowed_tools_env_set(self):
        """LOCI_POLICY_ALLOWED_TOOLS set → tool not in list is blocked (positive twin)."""
        env = {
            "LOCI_POLICY_ENABLE": "1",
            "LOCI_POLICY_ALLOWED_MODELS": "claude-haiku-4.5",
            "LOCI_POLICY_ALLOWED_TOOLS": "tool_a,tool_b",
        }
        with patch.dict(os.environ, env, clear=False):
            result = asyncio.run(loci_policy_hook(
                model="claude-haiku-4.5",
                messages=[],
                kwargs={"tool_call_name": "tool_c"},
                litellm_params={},
            ))
        self.assertEqual(result["reason"], "tool_not_allowed")
        self.assertFalse(result["apply"])
        violations = [v for v in result["violations"] if v["type"] == "tool_not_allowed"]
        self.assertEqual(len(violations), 1)


if __name__ == "__main__":
    unittest.main()
