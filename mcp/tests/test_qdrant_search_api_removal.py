"""QdrantClient.search() was removed in qdrant-client 1.16.

The pin in mcp/requirements.txt is >=1.17.0,<1.19.0, so on every supported
client the attribute lookup raised before a request was made — and both call
sites caught that in a bare except, turning a hard API break into "found
nothing". These tests use a client double that models the real surface: it has
query_points and NOT search, so anything reaching for .search fails.
"""
import json
import unittest
from unittest import mock

import server


class _Point:
    def __init__(self, pid, score, payload):
        self.id = pid
        self.score = score
        self.payload = payload


class _Response:
    def __init__(self, points):
        self.points = points


class _Client:
    """Models qdrant-client >=1.16: query_points exists, search does not."""

    def __init__(self, points, named_ok=True):
        self._points = points
        self._named_ok = named_ok
        self.calls = []

    def __getattr__(self, name):  # search must not be reachable
        raise AttributeError(f"'_Client' object has no attribute '{name}'")

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("using") == "dense" and not self._named_ok:
            raise ValueError("collection has a flat vector config")
        return _Response(list(self._points))


class _RaisingClient(_Client):
    def query_points(self, **kwargs):
        raise RuntimeError("qdrant is down")


def _patch(client, embed=(0.1, 0.2, 0.3)):
    return (
        mock.patch.object(server, "_get_qdrant", lambda: (client, "loci_memory")),
        mock.patch.object(server, "_embed", lambda _t: list(embed)),
    )


class TestDetectConflicts(unittest.TestCase):
    """Heuristic 1: a `gap` neighbour filled by an `observed` finding."""

    def _run(self, neighbour_payload, finding):
        client = _Client([_Point("n1", 0.9, neighbour_payload)])
        p1, p2 = _patch(client)
        with p1, p2:
            return client, server._detect_conflicts("inv-1", finding)

    def test_gap_filled_by_observation_is_detected(self):
        client, conflicts = self._run(
            {"id": "n1", "record_type": "gap", "text": "unknown whether the host phoned home"},
            {"id": "f1", "record_type": "observed", "text": "the host phoned home at 03:14"},
        )
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["neighbor_id"], "n1")
        self.assertEqual(conflicts[0]["neighbor_type"], "gap")

    def test_it_uses_query_points_and_keeps_the_score_gate(self):
        client, _ = self._run(
            {"id": "n1", "record_type": "gap", "text": "unknown"},
            {"id": "f1", "record_type": "observed", "text": "known"},
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["score_threshold"], 0.82)
        self.assertEqual(client.calls[0]["limit"], 10)

    def test_flat_collection_falls_back_from_the_named_vector(self):
        client = _Client(
            [_Point("n1", 0.9, {"id": "n1", "record_type": "gap", "text": "unknown"})],
            named_ok=False,
        )
        p1, p2 = _patch(client)
        with p1, p2:
            conflicts = server._detect_conflicts(
                "inv-1", {"id": "f1", "record_type": "observed", "text": "known"}
            )
        self.assertEqual(len(conflicts), 1)
        self.assertEqual([c.get("using") for c in client.calls], ["dense", None])

    def test_search_failure_still_fails_open(self):
        client = _RaisingClient([])
        p1, p2 = _patch(client)
        with p1, p2:
            self.assertEqual(
                server._detect_conflicts("inv-1", {"id": "f1", "record_type": "observed", "text": "x"}),
                [],
            )


class TestConflictNegationHeuristic(unittest.TestCase):
    """Heuristic 3 is a token-presence scan, so it is off unless asked for."""

    def _conflicts(self):
        client = _Client([_Point("n1", 0.9, {
            "id": "n1", "record_type": "observed",
            "text": "no adverse records were found for the account",
        })])
        p1, p2 = _patch(client)
        with p1, p2:
            return server._detect_conflicts(
                "inv-1", {"id": "f1", "record_type": "observed",
                          "text": "adverse records were found for the account"},
            )

    def test_off_by_default(self):
        with mock.patch.object(server, "_CONFLICT_NEGATION_HEURISTIC", False):
            self.assertEqual(self._conflicts(), [])

    def test_opt_in_restores_it(self):
        with mock.patch.object(server, "_CONFLICT_NEGATION_HEURISTIC", True):
            self.assertEqual(len(self._conflicts()), 1)


class TestMemoryConfidence(unittest.TestCase):
    def test_it_reports_a_trace_when_the_store_has_one(self):
        points = [
            _Point("p1", 0.81, {"text": "ryan was granted contractor access", "investigation_id": "i1", "confidence": "high"}),
            _Point("p2", 0.74, {"text": "contractor access was reviewed", "investigation_id": "i2", "confidence": "medium"}),
        ]
        client = _Client(points)
        p1, p2 = _patch(client)
        with p1, p2:
            payload = json.loads(server.memory_confidence("contractor access"))
        self.assertNotEqual(payload["basis"], "no_trace")
        self.assertGreater(payload["confidence"], 0.0)
        self.assertEqual(payload["cues"]["fluency"], 0.81)

    def test_a_broken_search_is_not_reported_as_no_trace(self):
        client = _RaisingClient([])
        p1, p2 = _patch(client)
        with p1, p2:
            payload = json.loads(server.memory_confidence("contractor access"))
        self.assertEqual(payload["basis"], "search_failed")
        self.assertEqual(payload["confidence"], 0.0)

    def test_an_empty_store_is_still_no_trace(self):
        client = _Client([])
        p1, p2 = _patch(client)
        with p1, p2:
            payload = json.loads(server.memory_confidence("contractor access"))
        self.assertEqual(payload["basis"], "no_trace")


if __name__ == "__main__":
    unittest.main()
