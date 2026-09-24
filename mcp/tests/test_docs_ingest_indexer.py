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
        (docs_dir / "one.md").write_text("""# One

This is the first doc.
""", encoding="utf-8")
        (docs_dir / "two.md").write_text("""# Two

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

    def test_docs_ingest_indexer_accepts_text_and_markdown_directory_extensions(self):
        docs_dir = Path(self._tmp.name) / "mixed_doc_sources"
        docs_dir.mkdir()
        (docs_dir / "guide.markdown").write_text("""# Markdown Guide

This doc should be backfilled.
""", encoding="utf-8")
        (docs_dir / "notes.txt").write_text("""# Notes

Text files should also be indexed.
""", encoding="utf-8")

        result = json.loads(
            server.docs_ingest_indexer(
                str(docs_dir),
                investigation_id="docs-ingest-text-extensions-test",
            )
        )
        self.assertEqual(result["stored"], 2, result)
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual({record["path"] for record in result["records"]}, {
            str(docs_dir / "guide.markdown"),
            str(docs_dir / "notes.txt"),
        })
        self.assertTrue(all(record["summary"] for record in result["records"]))

    def test_docs_ingest_indexer_skips_unchanged_docs(self):
        doc_path = Path(self._tmp.name) / "UNCHANGED.md"
        doc_path.write_text("""# Unchanged

The content stays stable.
""", encoding="utf-8")

        first = json.loads(
            server.docs_ingest_indexer(
                str(doc_path),
                investigation_id="docs-ingest-unchanged-test",
            )
        )
        second = json.loads(
            server.docs_ingest_indexer(
                str(doc_path),
                investigation_id="docs-ingest-unchanged-test",
            )
        )

        self.assertEqual(first["stored"], 1, first)
        self.assertEqual(second["stored"], 0, second)
        self.assertEqual(second["records"][0]["change_state"], "unchanged")
        self.assertEqual(second["records"][0]["changed"], False)

        loaded = json.loads(server.investigation_load("docs-ingest-unchanged-test"))
        self.assertEqual(len(loaded["recent_findings"]), 1)

    def test_docs_ingest_indexer_refreshes_changed_docs(self):
        doc_path = Path(self._tmp.name) / "CHANGED.md"
        doc_path.write_text("""# Original

This doc is about the old state.
""", encoding="utf-8")

        first = json.loads(
            server.docs_ingest_indexer(
                str(doc_path),
                investigation_id="docs-ingest-changed-test",
            )
        )
        doc_path.write_text("""# Revised

This doc is about the new state.
""", encoding="utf-8")
        second = json.loads(
            server.docs_ingest_indexer(
                str(doc_path),
                investigation_id="docs-ingest-changed-test",
            )
        )

        self.assertEqual(first["stored"], 1, first)
        self.assertEqual(second["stored"], 1, second)
        self.assertEqual(second["records"][0]["change_state"], "changed")
        self.assertEqual(second["records"][0]["changed"], True)

        loaded = json.loads(server.investigation_load("docs-ingest-changed-test"))
        self.assertEqual(len(loaded["recent_findings"]), 2)
        self.assertEqual(loaded["recent_findings"][0]["metadata"]["source_path"], str(doc_path))

    def test_docs_search_returns_matching_guidance(self):
        doc_path = Path(self._tmp.name) / "GUIDANCE.md"
        doc_path.write_text(
            """# Guidance

Different datasets are not interchangeable.

Claims must carry dataset, version, and scope.
""",
            encoding="utf-8",
        )

        server.docs_ingest_indexer(
            str(doc_path),
            investigation_id="docs-search-test",
        )

        result = json.loads(server.docs_search("dataset version scope", investigation_id="docs-search-test"))
        self.assertEqual(result["count"], 1, result)
        self.assertEqual(result["results"][0]["path"], str(doc_path))
        self.assertIn("dataset", result["results"][0]["summary"].lower())

    def test_docs_recall_returns_matching_guidance(self):
        doc_path = Path(self._tmp.name) / "DOCS_RECALL.md"
        doc_path.write_text(
            """# Guidance

Different datasets are not interchangeable.

Claims must carry dataset, version, and scope.
""",
            encoding="utf-8",
        )

        server.docs_ingest_indexer(
            str(doc_path),
            investigation_id="docs-recall-test",
        )

        result = json.loads(server.docs_recall("dataset version scope", investigation_id="docs-recall-test"))
        self.assertEqual(result["count"], 1, result)
        self.assertEqual(result["source"], "docs_search")
        self.assertEqual(result["results"][0]["path"], str(doc_path))
        self.assertIn("dataset", result["results"][0]["summary"].lower())

    def test_memory_surface_falls_back_to_docs_when_qdrant_unavailable(self):
        doc_path = Path(self._tmp.name) / "DOCS_SURFACE.md"
        doc_path.write_text(
            """# Surface Guidance

Different datasets are not interchangeable.

Claims must carry dataset, version, and scope.
""",
            encoding="utf-8",
        )

        server.docs_ingest_indexer(
            str(doc_path),
            investigation_id="docs-surface-fallback-test",
        )

        original_get_qdrant = server._get_qdrant
        server._get_qdrant = lambda: (None, None)
        try:
            result = json.loads(
                server.memory_surface(
                    "dataset version scope",
                    investigation_id="docs-surface-fallback-test",
                    top_k=5,
                )
            )
        finally:
            server._get_qdrant = original_get_qdrant

        self.assertEqual(result["count"], 1, result)
        self.assertEqual(result["surfaced"][0]["source"], "docs_search")
        self.assertIn("dataset", result["surfaced"][0]["text"].lower())

    def test_docs_search_reports_empty_or_no_match_query(self):
        result = json.loads(server.docs_search("", investigation_id="docs-search-empty-test"))
        self.assertEqual(result["count"], 0)
        self.assertIn("Query must be a non-empty string.", result["error"])

        doc_path = Path(self._tmp.name) / "EMPTY.md"
        doc_path.write_text("""# Only one idea

Everything here is unrelated to the searched phrase.
""", encoding="utf-8")
        server.docs_ingest_indexer(str(doc_path), investigation_id="docs-search-empty-test")

        result = json.loads(server.docs_search("dataset version scope", investigation_id="docs-search-empty-test"))
        self.assertEqual(result["count"], 0)
        self.assertIn("No matching docs found", result["error"])


if __name__ == "__main__":
    unittest.main()
