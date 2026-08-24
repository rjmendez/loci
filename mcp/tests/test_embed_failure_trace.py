"""Embedding failures fail open to [] — they must not do it silently.

An empty vector list is a load-bearing claim: every caller treats it as "no
embedding available" and degrades. Five different faults collapsed to the same
[] with nothing logged, so "the embedder is not configured", "the embedder is
down" and "the model returned the wrong count" were indistinguishable.
"""
import logging
import unittest
from unittest import mock

import embed_ops


class _Resp:
    def __init__(self, payload=None, boom=None):
        self._payload = payload or {}
        self._boom = boom

    def raise_for_status(self):
        if self._boom:
            raise self._boom

    def json(self):
        return self._payload


class TestEmbedFailureTrace(unittest.TestCase):
    def _run(self, resolve=("http://localhost:11434", "nomic-embed-text"), post=None):
        with mock.patch.object(embed_ops, "_resolve", lambda: resolve), \
             self.assertLogs("loci-mcp.embed_ops", level=logging.WARNING) as cm:
            if post is None:
                out = embed_ops.embed_texts(["a", "b"])
            else:
                with mock.patch("requests.post", post):
                    out = embed_ops.embed_texts(["a", "b"])
        return out, "\n".join(cm.output)

    def test_no_endpoint_configured_says_so(self):
        out, log = self._run(resolve=(None, "nomic-embed-text"))
        self.assertEqual(out, [])
        self.assertIn("no embedding endpoint configured", log)

    def test_a_short_batch_is_reported_not_swallowed(self):
        post = mock.Mock(return_value=_Resp({"embeddings": [[0.1]]}))   # 1 vector for 2 inputs
        out, log = self._run(post=post)
        self.assertEqual(out, [])
        self.assertIn("returned 1 vector(s) for 2 input(s)", log)

    def test_an_http_error_is_reported(self):
        post = mock.Mock(return_value=_Resp(boom=RuntimeError("500 Server Error")))
        out, log = self._run(post=post)
        self.assertEqual(out, [])
        self.assertIn("500 Server Error", log)

    def test_a_persistent_connection_error_is_reported_after_the_retry(self):
        import requests
        post = mock.Mock(side_effect=requests.exceptions.ConnectionError("refused"))
        out, log = self._run(post=post)
        self.assertEqual(out, [])
        self.assertIn("unreachable after retry", log)
        self.assertEqual(post.call_count, 2)   # cold-load retry preserved

    def test_the_happy_path_still_returns_vectors_and_logs_nothing(self):
        post = mock.Mock(return_value=_Resp({"embeddings": [[0.1], [0.2]]}))
        with mock.patch.object(embed_ops, "_resolve",
                               lambda: ("http://localhost:11434", "nomic-embed-text")), \
             mock.patch("requests.post", post), \
             self.assertNoLogs("loci-mcp.embed_ops", level=logging.WARNING):
            self.assertEqual(embed_ops.embed_texts(["a", "b"]), [[0.1], [0.2]])

    def test_an_empty_input_list_is_not_a_failure(self):
        with self.assertNoLogs("loci-mcp.embed_ops", level=logging.WARNING):
            self.assertEqual(embed_ops.embed_texts([]), [])


if __name__ == "__main__":
    unittest.main()
