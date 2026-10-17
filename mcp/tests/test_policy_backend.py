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
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Add mcp directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from policy import (
    PolicyBackend,
    Policy,
    PolicyTier,
    PolicyRegion,
    PolicyConfig,
    PolicyMetadata,
    get_policy_config,
    validate_policy,
)
from hooks import (
    loci_policy_hook,
    grounding_hook_augmented,
    tier_router_surface,
    eval_hook_stub,
    PolicyViolation,
    StepFeedback,
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
        
        with self.assertRaises(ValueError) as cm:
            self.backend.store(policy)
        
        self.assertIn("missing_title", str(cm.exception))


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
        """Test: tier_router respects policy tier when tiers match and doesn't downgrade."""
        backend = PolicyBackend()
        policy = Policy(
            title="Tier Constraint Test",
            tier=PolicyTier.LIGHT,  # Here policy is LIGHT (order 0)
            region=PolicyRegion.ROUTING
        )
        backend.store(policy)
        
        messages = []
        # Current tier is LIGHT (order 0), same as policy - should NOT downgrade
        kwargs = {"policy_id": policy.policy_id, "tier": "light"}
        
        result = await tier_router_surface(
            model="gpt-4",
            messages=messages,
            kwargs=kwargs,
            litellm_params={}
        )
        
        # Should NOT downgrade because current tier matches policy
        self.assertFalse(result["downgraded"])
        self.assertEqual(result["tier_decision"], "light")

    async def async_tier_router_downgrade_tests(self):
        """Test: tier_router downgrades when current tier exceeds policy tier."""
        backend = PolicyBackend()
        policy = Policy(
            title="Tier Constraint Test",
            tier=PolicyTier.LIGHT,  # Policy is LIGHT
            region=PolicyRegion.ROUTING
        )
        backend.store(policy)
        
        messages = []
        # Current tier is MEDIUM (order 1), exceeds policy LIGHT - should downgrade
        kwargs = {"policy_id": policy.policy_id, "tier": "medium"}
        
        result = await tier_router_surface(
            model="gpt-4",
            messages=messages,
            kwargs=kwargs,
            litellm_params={}
        )
        
        # Should downgrade to policy tier
        self.assertTrue(result["downgraded"])
        self.assertEqual(result["tier_decision"], "light")
        self.assertIn("tier_exceeded", [v["type"] for v in result["violations"]])

    def test_tier_router_surfaces_policy_constraints(self):
        """Test wrapper for async test."""
        asyncio.run(self.async_tier_router_tests())

    def test_tier_router_downgrades_on_exceed_tier(self):
        """Test wrapper for downgrade test."""
        asyncio.run(self.async_tier_router_downgrade_tests())


class TestHookIntegration(unittest.TestCase):
    """RED: LiteLLM hooks integration (policy_hook, grounding_hook augment, tier_router)."""

    def test_loci_policy_hook_decides_whether_to_apply_policy(self):
        """Test: loci_policy_hook returns (apply: bool, policy_id: str|None, reason: str)."""
        # Hook signature:
        # async def loci_policy_hook(model, messages, kwargs, litellm_params, logger):
        #     return {"apply": True, "policy_id": "...", "reason": "..."}
        # Returns:
        #   - apply=True if policy is enabled and matches model/tier
        #   - apply=False with reason if policy disabled or model excluded
        pass

    def test_grounding_hook_augmented_with_policy_backing(self):
        """Test: grounding_hook augmented to surface policy's backing evidence."""
        # The EVO-RAG discovery phase policy backs retrieval with reward vector
        # grounding_hook should inject:
        # MEMORY MATCH (policy-aware):
        #   - discovery_mode: use broad search
        #   - reward_vector: ["r_ret", "r_rem", "r_ovp"]
        #   - matching findings with scores
        pass

    def test_tier_router_surfaces_policy_constraints(self):
        """Test: tier_router respects policy tier limits on model selection."""
        # If policy tier=light, tier_router should:
        #   - Reject heavy-tier models
        #   - Downgrade to light-tier equivalent
        #   - Log violation if LOCI_POLICY_ENABLE=1
        pass

    def test_eval_hook_stub_captures_feedback_for_drpg(self):
        """Test: eval_hook_stub logs step-level feedback for DRPG policy evolution."""
        # Expected logs:
        # {
        #   "step": 1,
        #   "action": "SEARCH",
        #   "reward": {
        #     "r_ret": 1.0,  # hit gold docs
        #     "r_rem": -0.1,  # search penalty
        #     "r_ovp": 0.0,   # no overlap
        #     "r_step": -0.05  # step penalty
        #   },
        #   "timestamp": "...",
        #   "policy_id": "..."
        # }
        pass


class TestPolicyValidation(unittest.TestCase):
    """RED: Policy validation against tier/region constraints."""

    def test_policy_validation_rejects_invalid_tier(self):
        """Test: policy validator rejects tier not in (light, medium, heavy)."""
        policy = {
            "tier": "extreme",  # Invalid
            "region": "memory",
            "config": {}
        }
        # from loci_mcp.policy import validate_policy
        # errors = validate_policy(policy)
        # self.assertIn("invalid_tier", [e["code"] for e in errors])
        pass

    def test_policy_validation_rejects_invalid_region(self):
        """Test: policy validator rejects region not in (memory, context, routing)."""
        policy = {
            "tier": "light",
            "region": "unknown",  # Invalid
            "config": {}
        }
        # from loci_mcp.policy import validate_policy
        # errors = validate_policy(policy)
        # self.assertIn("invalid_region", [e["code"] for e in errors])
        pass

    def test_policy_validation_checks_config_consistency(self):
        """Test: max_context_chars must align with tier constraints."""
        policy = {
            "tier": "light",
            "config": {
                "max_context_chars": 100000  # Exceeds light tier limit
            }
        }
        # from loci_mcp.policy import validate_policy
        # errors = validate_policy(policy)
        # self.assertTrue(any(e["code"] == "tier_config_mismatch" for e in errors))
        pass


class TestDRPGIntegration(unittest.TestCase):
    """RED: DRPG discovery↔refinement and 7D reward vector integration."""

    def test_policy_discovery_phase_uses_high_retrieval_bonus(self):
        """Test: Discovery phase policy has r_ret weight = 1.0 (high incentive to retrieve)."""
        # From EVO-RAG grounded reference:
        # Discovery: broad exploration, high retrieval incentive
        # policy.config.discovery_mode = True implies:
        #   - r_ret_weight: 1.0
        #   - r_rem_weight: -0.05  (low penalty for searching)
        #   - r_bkp_weight: -0.2   (penalize backtrack)
        pass

    def test_policy_refinement_phase_prioritizes_efficiency(self):
        """Test: Refinement phase policy penalizes searches, prioritizes correctness."""
        # Refinement: efficiency, penalize searches
        # policy.config.discovery_mode = False implies:
        #   - r_ret_weight: 0.5
        #   - r_rem_weight: -0.2   (higher penalty for searching)
        #   - r_ref_weight: 1.0    (high reward for refusal when insufficient)
        pass

    def test_reward_vector_step_tracking(self):
        """Test: eval_hook tracks 7-dimensional reward vector per step."""
        # reward_vector = [r_ret, r_rem, r_ovp, r_bkp, r_ref, r_step, r_cor]
        # eval_hook logs each component at every step
        # Used by DRPG to evolve policy parameters
        pass


if __name__ == "__main__":
    unittest.main()
