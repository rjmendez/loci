"""Causal LLM edges must name findings that exist.

The prompt numbers the finding list as "{idx+1}. [{id}] ...", and the model
sometimes returns the ORDINAL instead of the id. Measured on a live run: three
edges written with source_id "1", "2", "3". The guard checked only that the
strings were non-empty, and causal_edges_list strips `method`, so a consumer
could not tell a dangling edge from a real one.

_declared_causal_edges already validates against the finding set; the LLM path
did not.
"""
import unittest
from unittest import mock

import server


class CausalLlmIdValidationTest(unittest.TestCase):

    def _run(self, llm_reply, findings):
        """Patch the ATTRIBUTE on the memcheck package, not sys.modules.

        _run_causal_inference does `from memcheck import llm`, which reads
        memcheck.llm as an attribute. Patching sys.modules["memcheck.llm"] only
        takes effect if the package has not been imported yet, so the first
        version of this test passed alone and failed inside the full suite —
        order-dependent, which is no test at all.
        """
        import memcheck
        fake = mock.MagicMock()
        fake.llm_available.return_value = True
        fake.call_llm.return_value = llm_reply
        with mock.patch.object(memcheck, "llm", fake, create=True), \
             mock.patch.dict("sys.modules", {"memcheck.llm": fake}), \
             mock.patch.object(server, "_append_jsonl") as appended, \
             mock.patch.object(server, "MEMORY_DIR"):
            server._run_causal_inference("inv", findings)
        return [c.args[1] for c in appended.call_args_list]

    def test_ordinal_ids_are_dropped(self):
        """The measured corruption: source_id '1' instead of a uuid."""
        findings = [{"id": "aaa", "text": "one"}, {"id": "bbb", "text": "two"}]
        reply = '{"source_id": "1", "target_id": "2", "edge_type": "caused_by", "confidence": 0.9}'
        written = self._run(reply, findings)
        llm_edges = [e for e in written if e.get("method") == "llm_slow_path"]
        self.assertEqual(llm_edges, [], "an edge naming ordinals must not be written")

    def test_real_ids_are_kept(self):
        findings = [{"id": "aaa", "text": "one"}, {"id": "bbb", "text": "two"}]
        reply = '{"source_id": "aaa", "target_id": "bbb", "edge_type": "caused_by", "confidence": 0.9}'
        written = self._run(reply, findings)
        llm_edges = [e for e in written if e.get("method") == "llm_slow_path"]
        self.assertEqual(len(llm_edges), 1)
        self.assertEqual((llm_edges[0]["source_id"], llm_edges[0]["target_id"]), ("aaa", "bbb"))

    def test_a_half_valid_edge_is_dropped(self):
        """One real id and one ordinal is still a dangling edge."""
        findings = [{"id": "aaa", "text": "one"}, {"id": "bbb", "text": "two"}]
        reply = '{"source_id": "aaa", "target_id": "2", "edge_type": "caused_by", "confidence": 0.9}'
        written = self._run(reply, findings)
        self.assertEqual([e for e in written if e.get("method") == "llm_slow_path"], [])


if __name__ == "__main__":
    unittest.main()
