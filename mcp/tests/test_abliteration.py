import json
import unittest
from unittest import mock

import abliteration


class _Resp:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, response):
        self.response = response

    def post(self, *_args, **_kwargs):
        return self.response


class TestAbliteration(unittest.TestCase):
    def test_no_key_degrades_without_raise(self):
        with mock.patch.object(abliteration, "credentials", lambda: ("https://x", "")):
            out = abliteration.generate_batch(["p1", "p2"])
        self.assertEqual(out, [{"text": "", "ok": False}, {"text": "", "ok": False}])

    def test_fmt_json_extracts_nested_object(self):
        resp = _Resp({
            "choices": [{"message": {"content": "thinking...\n{\"a\":1}"}}],
            "model": "abliterated-model",
            "usage": {},
        })
        with mock.patch.object(abliteration, "credentials", lambda: ("https://x", "k")):
            out = abliteration.generate_batch(["p"], fmt="json", session_fn=lambda: _Session(resp))
        self.assertTrue(out[0]["ok"])
        self.assertEqual(out[0]["text"], "{\"a\":1}")

    def test_guardrail_blocks_excessive_max_tokens(self):
        with mock.patch.dict("os.environ", {"LOCI_ABLITERATION_MAX_TOKENS_PER_CALL": "12"}), \
             mock.patch.object(abliteration, "credentials", lambda: ("https://x", "k")):
            out = abliteration.generate_batch(["p"], max_tokens=64)
        self.assertFalse(out[0]["ok"])
        self.assertIn("cap exceeded", out[0].get("why", ""))


if __name__ == "__main__":
    unittest.main()
