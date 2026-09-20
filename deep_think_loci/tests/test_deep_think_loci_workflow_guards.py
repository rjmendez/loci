"""Static guardrail checks for deep-think-loci workflow safety.

These tests validate that high-risk safety guards remain present in the JS
workflow script without requiring the Workflow runtime.
"""
from pathlib import Path
import unittest


WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "deep-think-loci.js"


class DeepThinkLociWorkflowGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_rejects_invalid_json_args(self):
        self.assertIn("args must be valid JSON object text", self.text)
        self.assertIn("parseWorkflowArgs(rawArgs)", self.text)

    def test_enforces_numeric_bounds(self):
        self.assertIn("ideas_per_agent', 10, { min: 1, max: 25 }", self.text)
        self.assertIn("ground_threshold', 0.59, { min: 0.5, max: 0.95 }", self.text)

    def test_enforces_target_and_path_safety(self):
        self.assertIn("TARGET_NAME_RE = /^[A-Za-z0-9_-]{1,64}$/", self.text)
        self.assertIn("normalizeScratchRoot", self.text)
        self.assertIn("resolved.includes('..')", self.text)
        self.assertIn("resolved.includes('//')", self.text)

    def test_ideation_phase_is_grounding_gated(self):
        self.assertIn("Ideation generator a${i + 1}", self.text)
        self.assertIn("2) ${groundBlock(t.focus", self.text)

    def test_legacy_unsafe_half_tmp_paths_removed(self):
        self.assertNotIn("/tmp/half", self.text)
        self.assertIn("mkdir -p ${shellQuote(SCRATCH_ROOT)}", self.text)


if __name__ == "__main__":
    unittest.main()
