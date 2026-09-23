import json
import sys
import tempfile
import unittest
from pathlib import Path

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import server  # noqa: E402


class DocsIngestIndexerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_docs_ingest_indexer_stores_provenanced_summary(self):
        doc_path = Path(self._tmp.name) / "FLYBRAIN_GUIDE.md"
        doc_path.write_text(
            """# FlyBrain Guide

Different datasets are not interchangeable.

Claims must carry dataset, version, and scope.
""",
            encoding="utf-8",
        )

        result = json.loads(
            server.docs_ingest_indexer(
                str(doc_path),
                investigation_id="docs-ingest-indexer-test",
                summary_only=True,
            )
        )
        self.assertEqual(result["stored"], 1, result)
        self.assertEqual(len(result["records"]), 1)

        loaded = json.loads(server.investigation_load("docs-ingest-indexer-test"))
        finding = loaded["recent_findings"][0]
        self.assertTrue(finding["metadata"]["doc_summary"])
        self.assertEqual(finding["metadata"]["source_path"], str(doc_path))
        self.assertEqual(finding["metadata"]["provenance"]["tool_name"], "docs_ingest_indexer")
        self.assertTrue(finding["metadata"]["source_sha256"])

    def test_docs_ingest_indexer_accepts_directory_input(self):
        docs_dir = Path(self._tmp.name) / "flybrain_docs"
        docs_dir.mkdir()
        (docs_dir / "one.markdown").write_text("""# One

This is the first doc.
""", encoding="utf-8")
        (docs_dir / "two.txt").write_text("""# Two

This is the second doc.
""", encoding="utf-8")

        result = json.loads(
            server.docs_ingest_indexer(
                str(docs_dir),
                investigation_id="docs-ingest-directory-test",
            )
        )
        self.assertEqual(result["stored"], 2, result)
        self.assertEqual(len(result["records"]), 2)

        loaded = json.loads(server.investigation_load("docs-ingest-directory-test"))
        self.assertEqual(len(loaded["recent_findings"]), 2)
        self.assertEqual(
            {finding["metadata"]["source_path"] for finding in loaded["recent_findings"]},
            {str(docs_dir / "one.markdown"), str(docs_dir / "two.txt")},
        )
        self.assertEqual(
            {finding["metadata"]["provenance"]["tool_name"] for finding in loaded["recent_findings"]},
            {"docs_ingest_indexer"},
        )

    def test_docs_ingest_indexer_rejects_symlinked_directory_root(self):
        external = Path(self._tmp.name) / "external_docs"
        external.mkdir()
        (external / "outside.md").write_text("""# Outside

This should not be traversed through a symlinked ingest root.
""", encoding="utf-8")
        docs_dir = Path(self._tmp.name) / "flybrain_docs_link"
        docs_dir.symlink_to(external, target_is_directory=True)

        result = json.loads(
            server.docs_ingest_indexer(
                str(docs_dir),
                investigation_id="docs-ingest-symlink-test",
            )
        )
        self.assertEqual(result["error"], f"No readable markdown/text documents found under: {docs_dir}", result)
        self.assertEqual(result["stored"], 0, result)


if __name__ == "__main__":
    unittest.main()
