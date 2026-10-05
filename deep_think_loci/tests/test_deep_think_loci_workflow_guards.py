"""Guardrail checks for deep-think-loci workflow safety.

The argument guards are EXECUTED in node: the workflow's parameter block runs
against stub runtime globals (phase/agent/parallel) and stops at the second
agent call, so a removed or loosened bounds check changes what these tests see.
The remaining text checks cover prompt wording that has no behaviour to run.
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest


WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "deep-think-loci.js"

# Stub runtime: record prompts, stop after the Ideate prompt (which carries the
# resolved parameters), and print either the prompts or the validation error.
# The workflow file is read as-is; only `export` is dropped from its meta line.
_DRIVER = r"""
import { readFileSync } from 'node:fs'
// The Workflow runtime runs the script body as an async function (it ends in a
// top-level return), with args/phase/agent/parallel in scope; do the same.
const src = readFileSync(process.argv[2], 'utf8').replace(/^export const meta/m, 'const meta')
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor
const body = new AsyncFunction('args', 'phase', 'agent', 'parallel', src)
const prompts = []
const agent = async (p) => { prompts.push(p); if (prompts.length >= 2) throw new Error('STOP'); return 'ok' }
const parallel = async (fns) => Promise.all(fns.map((f) => f()))
try {
  await body(JSON.parse(process.argv[3]), () => {}, agent, parallel)
} catch (e) {
  if (e.message !== 'STOP') { console.log(JSON.stringify({ error: e.message })); process.exit(0) }
}
console.log(JSON.stringify({ prompts }))
"""


def run_params(args: dict) -> dict:
    """Run the workflow's parameter block in node with ``args``."""
    with tempfile.TemporaryDirectory() as tmp:
        drv = Path(tmp) / "driver.mjs"
        drv.write_text(_DRIVER, encoding="utf-8")
        out = subprocess.run(["node", str(drv), str(WORKFLOW), json.dumps(args)],
                             capture_output=True, text=True, timeout=60, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


@unittest.skipIf(shutil.which("node") is None, "node is not installed")
class DeepThinkLociArgumentGuardTests(unittest.TestCase):
    def test_defaults_resolve(self):
        out = run_params({})
        self.assertNotIn("error", out)
        ideate = out["prompts"][1]
        self.assertIn("Return exactly 10 concrete one-line improvement ideas", ideate)
        self.assertIn("--threshold 0.59", ideate)

    def test_integer_bounds_are_enforced_inclusively(self):
        for ok in (1, 25):
            out = run_params({"ideas_per_agent": ok})
            self.assertIn(f"Return exactly {ok} concrete", out["prompts"][1])
        for bad in (0, 26, 2.5, "10"):
            out = run_params({"ideas_per_agent": bad})
            self.assertIn("invalid ideas_per_agent: expected an integer in [1, 25]", out.get("error", ""), bad)

    def test_number_bounds_are_enforced_inclusively(self):
        for ok in (0.5, 0.95):
            self.assertIn(f"--threshold {ok}", run_params({"ground_threshold": ok})["prompts"][1])
        for bad in (0.49, 0.96, "0.7"):
            out = run_params({"ground_threshold": bad})
            self.assertIn("invalid ground_threshold: expected a finite number in [0.5, 0.95]",
                          out.get("error", ""), bad)

    def test_target_names_and_scratch_root_are_validated(self):
        bad_target = run_params({"targets": [{"name": "../etc", "focus": "x"}]})
        self.assertIn("invalid targets[0].name", bad_target.get("error", ""))
        for root in ("/tmp/../etc", "/tmp//x", "relative/path", "/tmp/a b"):
            self.assertIn("invalid scratch_root", run_params({"scratch_root": root}).get("error", ""), root)
        ok = run_params({"targets": [{"name": "t-1", "focus": "the focus"}], "scratch_root": "/tmp/dt/"})
        self.assertNotIn("error", ok)
        self.assertIn("/tmp/dt/ideate_a1_t-1_cand.json", ok["prompts"][1])

    def test_invalid_json_args_are_rejected(self):
        # args arrive as text from the Workflow tool; a non-object must not be accepted
        out = run_params("not json")
        self.assertIn("args must be valid JSON object text", out.get("error", ""))


class DeepThinkLociWorkflowGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_ideation_phase_is_grounding_gated(self):
        self.assertIn("Ideation generator a${i + 1}", self.text)
        self.assertIn("2) ${groundBlock(t.focus", self.text)

    def test_legacy_unsafe_half_tmp_paths_removed(self):
        self.assertNotIn("/tmp/half", self.text)
        self.assertIn("mkdir -p ${shellQuote(SCRATCH_ROOT)}", self.text)


if __name__ == "__main__":
    unittest.main()
