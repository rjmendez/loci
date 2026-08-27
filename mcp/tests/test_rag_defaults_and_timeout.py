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
        with mock.patch.dict(os.environ, {"LOCI_QDRANT_TIMEOUT": "42"}):
            import qdrant_ops
            importlib.reload(qdrant_ops)
            try:
                self.assertEqual(qdrant_ops._QDRANT_TIMEOUT, 42.0)
            finally:
                importlib.reload(qdrant_ops)


class QueryExpandShadowTest(unittest.TestCase):

    def test_the_cli_wrapper_reexports_expand(self):
        """Whichever module wins the name, expand() must exist."""
        import importlib.util
        import sys
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "..", "scripts", "query_expand.py")
        spec = importlib.util.spec_from_file_location("scripts_query_expand", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["scripts_query_expand"] = mod
        try:
            spec.loader.exec_module(mod)
            self.assertTrue(hasattr(mod, "expand"),
                            "the CLI wrapper must re-export expand() or it shadows it away")
            self.assertIsNotNone(mod.expand)
        finally:
            sys.modules.pop("scripts_query_expand", None)


class RagDocstringTest(unittest.TestCase):

    def test_docstring_does_not_claim_agent_core_chunks_is_a_default(self):
        import server
        doc = server.rag_context_search.__doc__ or ""
        self.assertIn("CODE_CHUNKS_COLLECTION", doc)
        self.assertNotIn('default: ["loci_memory", "agent_core_chunks"]', doc)


if __name__ == "__main__":
    unittest.main()
