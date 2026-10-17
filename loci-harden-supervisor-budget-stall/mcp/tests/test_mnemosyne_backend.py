"""Tests for MnemosyneBackend defensive filtering.

No live Qdrant/mnemosyne connection required — all callables are injected
via unittest.mock.
"""

import asyncio
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock

_MCP_DIR = str(Path(__file__).resolve().parent.parent)
if _MCP_DIR not in sys.path:
    sys.path.insert(0, _MCP_DIR)

from memcheck.mnemosyne import MnemosyneBackend, _extract_metadata  # noqa: E402
from memcheck.verdict import new_verdict  # noqa: E402


def _make_verdict(kind: str = "memory") -> object:
    return new_verdict(
        subject_kind=kind,
        subject_signature="sig-" + uuid.uuid4().hex[:8],
        subject_excerpt="test excerpt",
        verdict_type="contradiction",
        decision="flag",
        confidence=0.9,
        rationale="test",
        source="rule",
    )


def _valid_result(verdict_obj, score: float = 0.85) -> dict:
    """Build a flat mnemosyne result envelope containing a memcheck verdict."""
    payload = verdict_obj.to_payload()
    payload["memcheck"] = True
    return {"metadata": payload, "score": score}


class TestExtractMetadata(unittest.TestCase):
    """_extract_metadata() handles all four envelope shapes."""

    def test_flat_metadata_key(self):
        meta = {"answer": 42}
        result = {"metadata": meta}
        self.assertEqual(_extract_metadata(result), meta)

    def test_memory_wrapper(self):
        meta = {"answer": 1}
        result = {"memory": {"metadata": meta}}
        self.assertEqual(_extract_metadata(result), meta)

    def test_item_wrapper(self):
        meta = {"answer": 2}
        result = {"item": {"metadata": meta}}
        self.assertEqual(_extract_metadata(result), meta)

    def test_payload_wrapper(self):
        meta = {"answer": 3}
        result = {"payload": {"metadata": meta}}
        self.assertEqual(_extract_metadata(result), meta)

    def test_data_wrapper(self):
        meta = {"answer": 4}
        result = {"data": {"metadata": meta}}
        self.assertEqual(_extract_metadata(result), meta)

    def test_non_dict_returns_empty(self):
        self.assertEqual(_extract_metadata("not a dict"), {})
        self.assertEqual(_extract_metadata(None), {})
        self.assertEqual(_extract_metadata([]), {})

    def test_missing_metadata_returns_empty(self):
        self.assertEqual(_extract_metadata({}), {})

    def test_non_dict_nested_metadata_falls_through(self):
        # metadata is a string, not a dict → should check fallback keys
        result = {"metadata": "oops", "memory": {"metadata": {"real": True}}}
        self.assertEqual(_extract_metadata(result), {"real": True})


class TestMemcheckFiltering(unittest.IsolatedAsyncioTestCase):
    """recall() drops results where memcheck != True."""

    async def test_memcheck_true_passes(self):
        v = _make_verdict("memory")
        recall_mock = MagicMock(return_value=[_valid_result(v)])
        backend = MnemosyneBackend(recall=recall_mock)
        results = await backend.recall("query", [], "memory", top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].verdict.subject_kind, "memory")

    async def test_memcheck_missing_filtered(self):
        v = _make_verdict("memory")
        payload = v.to_payload()  # no memcheck key
        recall_mock = MagicMock(return_value=[{"metadata": payload, "score": 0.9}])
        backend = MnemosyneBackend(recall=recall_mock)
        results = await backend.recall("query", [], "memory", top_k=5)
        self.assertEqual(results, [])

    async def test_memcheck_false_filtered(self):
        v = _make_verdict("memory")
        payload = {**v.to_payload(), "memcheck": False}
        recall_mock = MagicMock(return_value=[{"metadata": payload, "score": 0.9}])
        backend = MnemosyneBackend(recall=recall_mock)
        results = await backend.recall("query", [], "memory", top_k=5)
        self.assertEqual(results, [])


class TestSubjectKindFiltering(unittest.IsolatedAsyncioTestCase):
    """recall() keeps only results matching the requested kind."""

    async def test_matching_kind_returned(self):
        v = _make_verdict("action")
        recall_mock = MagicMock(return_value=[_valid_result(v)])
        backend = MnemosyneBackend(recall=recall_mock)
        results = await backend.recall("query", [], "action", top_k=5)
        self.assertEqual(len(results), 1)

    async def test_wrong_kind_filtered(self):
        v = _make_verdict("output")
        recall_mock = MagicMock(return_value=[_valid_result(v)])
        backend = MnemosyneBackend(recall=recall_mock)
        results = await backend.recall("query", [], "memory", top_k=5)
        self.assertEqual(results, [])

    async def test_mixed_kinds_only_matching_returned(self):
        v_memory = _make_verdict("memory")
        v_action = _make_verdict("action")
        recall_mock = MagicMock(
            return_value=[_valid_result(v_memory), _valid_result(v_action)]
        )
        backend = MnemosyneBackend(recall=recall_mock)
        results = await backend.recall("query", [], "memory", top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].verdict.subject_kind, "memory")


class TestMalformedPayloadTolerance(unittest.IsolatedAsyncioTestCase):
    """recall() skips malformed payloads without raising."""

    async def test_missing_required_field_skipped(self):
        bad = {"memcheck": True, "subject_kind": "memory"}  # missing id etc.
        recall_mock = MagicMock(return_value=[{"metadata": bad, "score": 0.5}])
        backend = MnemosyneBackend(recall=recall_mock)
        results = await backend.recall("query", [], "memory", top_k=5)
        self.assertEqual(results, [])

    async def test_completely_empty_metadata_skipped(self):
        recall_mock = MagicMock(return_value=[{"metadata": {}, "score": 0.5}])
        backend = MnemosyneBackend(recall=recall_mock)
        results = await backend.recall("query", [], "memory", top_k=5)
        self.assertEqual(results, [])

    async def test_non_dict_result_skipped(self):
        recall_mock = MagicMock(return_value=["not a dict", None, 42])
        backend = MnemosyneBackend(recall=recall_mock)
        results = await backend.recall("query", [], "memory", top_k=5)
        self.assertEqual(results, [])

    async def test_good_and_bad_mixed(self):
        v = _make_verdict("memory")
        recall_mock = MagicMock(
            return_value=[
                {"metadata": {"memcheck": True}, "score": 0.9},  # bad: no kind/id
                _valid_result(v),
            ]
        )
        backend = MnemosyneBackend(recall=recall_mock)
        results = await backend.recall("query", [], "memory", top_k=5)
        self.assertEqual(len(results), 1)


class TestForgetNoOp(unittest.IsolatedAsyncioTestCase):
    """forget() always returns 0 — mnemosyne has no delete tool."""

    async def test_forget_returns_zero(self):
        backend = MnemosyneBackend()
        result = await backend.forget("some text", "memory")
        self.assertEqual(result, 0)

    async def test_forget_with_recall_injected_still_returns_zero(self):
        recall_mock = MagicMock(return_value=[])
        backend = MnemosyneBackend(recall=recall_mock)
        result = await backend.forget("text", "action")
        self.assertEqual(result, 0)


class TestNoneCallableSafetyGuard(unittest.IsolatedAsyncioTestCase):
    """None remember/recall callables are safe no-ops."""

    async def test_recall_none_returns_empty(self):
        backend = MnemosyneBackend(recall=None)
        results = await backend.recall("query", [], "memory", top_k=5)
        self.assertEqual(results, [])

    async def test_record_none_does_not_raise(self):
        backend = MnemosyneBackend(remember=None)
        v = _make_verdict("memory")
        result = await backend.record(v)  # should not raise
        self.assertIsNone(result)

    async def test_both_none_recall_empty(self):
        backend = MnemosyneBackend()
        results = await backend.recall("query", [], "memory", top_k=5)
        self.assertEqual(results, [])

    async def test_empty_results_from_recall_returns_empty(self):
        recall_mock = MagicMock(return_value=[])
        backend = MnemosyneBackend(recall=recall_mock)
        results = await backend.recall("query", [], "memory", top_k=5)
        self.assertEqual(results, [])

    async def test_none_results_from_recall_returns_empty(self):
        recall_mock = MagicMock(return_value=None)
        backend = MnemosyneBackend(recall=recall_mock)
        results = await backend.recall("query", [], "memory", top_k=5)
        self.assertEqual(results, [])

    async def test_record_calls_remember(self):
        remember_mock = MagicMock(return_value=True)
        backend = MnemosyneBackend(remember=remember_mock)
        v = _make_verdict("memory")
        await backend.record(v)
        remember_mock.assert_called_once()
        _, kwargs = remember_mock.call_args
        self.assertTrue(kwargs["metadata"].get("memcheck"))


if __name__ == "__main__":
    unittest.main()
