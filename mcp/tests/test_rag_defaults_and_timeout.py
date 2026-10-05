"""Three defects a corpus audit measured against the live instance.

1. The Qdrant client timeout was 5s. Measured on agent_core_chunks (6.06M points,
   unquantized): 1.31s filtered, 10.87s unfiltered, 9.01s on a doc_type filter. So
   every explicit query to it returned mode="rag_failed", result_count=0 at HTTP 200.

2. rag_context_search's docstring claimed it searched agent_core_chunks by default.
   It never did — defaults come from QDRANT_COLLECTION_PREFIX + CODE_CHUNKS_COLLECTION.
   The docstring sent callers to override with a collection that is 86% GPS
   trajectory points and 0.18% code.

3. scripts/query_expand.py shares a module NAME with mcp/query_expand.py but defined
   only main(). scripts/judge_eval.py puts scripts/ AHEAD of mcp/ on sys.path, so
   `import query_expand` could resolve to the wrapper, leaving _rag_expand_queries to
   fail open with "module 'query_expand' has no attribute 'expand'" — silently
   disabling the expansion whose +4% nDCG@10 is the reason it exists.
"""
import importlib
import os
import unittest
from unittest import mock


class QdrantTimeoutTest(unittest.TestCase):

    def test_default_timeout_exceeds_the_measured_slow_search(self):
        import qdrant_ops
        self.assertGreaterEqual(
            qdrant_ops._QDRANT_TIMEOUT, 11.0,
            "10.87s was measured on agent_core_chunks; a shorter timeout fails it")

    def test_timeout_is_configurable(self):
        import qdrant_ops
        before = qdrant_ops._QDRANT_TIMEOUT
        try:
            with mock.patch.dict(os.environ, {"LOCI_QDRANT_TIMEOUT": "42"}):
                importlib.reload(qdrant_ops)
                self.assertEqual(qdrant_ops._QDRANT_TIMEOUT, 42.0)
        finally:
            # Outside the env patch: reloading inside it re-read "42" and leaked
            # 42.0 into every later test in the session.
            importlib.reload(qdrant_ops)
        self.assertEqual(qdrant_ops._QDRANT_TIMEOUT, before)


_MCP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(os.path.dirname(_MCP_DIR), "scripts")


class QueryExpandShadowTest(unittest.TestCase):

    def _import_query_expand(self, path_front):
        """Import `query_expand` in a fresh interpreter with ``path_front`` first
        on sys.path, and report what the name resolved to and what expand() does
        with a stub generator (so no model is needed)."""
        import json
        import subprocess
        import sys
        import textwrap
        code = textwrap.dedent("""
            import json, sys
            sys.path[:0] = json.loads(sys.argv[1])
            import query_expand
            expand = getattr(query_expand, "expand", None)
            out = {"callable": callable(expand)}
            if callable(expand):
                out["defined_in"] = expand.__code__.co_filename
                gen = lambda prompt, **kw: {"ok": True, "text": json.dumps(
                    {"queries": ["alpha rewrite"], "keywords": ["beta"]})}
                out["result"] = expand("alpha", gen_fn=gen, n_queries=2, n_keywords=2)
            print(json.dumps(out))
        """)
        proc = subprocess.run([sys.executable, "-c", code, json.dumps(path_front)],
                              capture_output=True, text=True, timeout=60, cwd=_MCP_DIR)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_scripts_ahead_of_mcp_still_resolves_the_real_expand(self):
        # scripts/judge_eval.py leaves scripts/ AHEAD of mcp/ on sys.path, and the
        # server then does `import query_expand`. Reproduce that order exactly.
        got = self._import_query_expand([_SCRIPTS_DIR, _MCP_DIR])
        self.assertTrue(got["callable"], "scripts/query_expand.py shadowed expand() away")
        self.assertEqual(os.path.realpath(got["defined_in"]),
                         os.path.realpath(os.path.join(_MCP_DIR, "query_expand.py")))
        self.assertEqual(got["result"]["degraded"], False)
        self.assertEqual(got["result"]["queries"][0], "alpha")
        self.assertIn("alpha rewrite", got["result"]["queries"])

    def test_mcp_first_resolves_the_real_expand(self):
        got = self._import_query_expand([_MCP_DIR, _SCRIPTS_DIR])
        self.assertEqual(os.path.realpath(got["defined_in"]),
                         os.path.realpath(os.path.join(_MCP_DIR, "query_expand.py")))
        self.assertEqual(got["result"]["degraded"], False)


class RagDocstringTest(unittest.TestCase):

    def test_docstring_does_not_claim_agent_core_chunks_is_a_default(self):
        import server
        doc = server.rag_context_search.__doc__ or ""
        self.assertIn("CODE_CHUNKS_COLLECTION", doc)
        self.assertNotIn('default: ["loci_memory", "agent_core_chunks"]', doc)


if __name__ == "__main__":
    unittest.main()
