"""The OpenRouter tier.

Two behaviours carry most of the weight here. OpenRouter reports upstream
failures as **HTTP 200 with an `error` object**, so a status-code check alone
turns a rate-limit into something indistinguishable from a model that answered
with nothing — which is the exact failure shape this codebase keeps finding
elsewhere. And the free pool is a shared upstream quota, so 429 is its normal
state under load and the ladder has to treat it as a hint, not an outage.
"""
import json
import unittest
from unittest import mock

import openrouter


class _Resp:
    def __init__(self, payload, status=200, raw=None):
        self.status_code = status
        self._payload = payload
        self.text = raw if raw is not None else json.dumps(payload)

    def json(self):
        if self._payload is _BAD:
            raise ValueError("not json")
        return self._payload


_BAD = object()


def _ok(content, model="m", usage=None):
    return {"choices": [{"message": {"content": content}}],
            "model": model, "usage": usage or {}}


class _Session:
    """Answers per model name, so ladder behaviour is observable."""

    def __init__(self, by_model, default=None):
        self.by_model = by_model
        self.default = default
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        model = (json or {}).get("model")
        self.calls.append(model)
        resp = self.by_model.get(model, self.default)
        if isinstance(resp, Exception):
            raise resp
        return resp


def _creds():
    return mock.patch.object(openrouter, "credentials", lambda: ("https://x/api/v1", "k"))


class TestErrorEnvelope(unittest.TestCase):
    def test_a_200_carrying_an_error_object_is_a_failure(self):
        sess = _Session({}, default=_Resp(
            {"error": {"code": 429, "message": "temporarily rate-limited upstream"}}))
        with _creds():
            out = openrouter.generate_batch(["p"], model="m", session_fn=lambda: sess)
        self.assertFalse(out[0]["ok"])
        self.assertEqual(out[0]["status"], 429)

    def test_a_real_non_200_is_still_a_failure(self):
        sess = _Session({}, default=_Resp({"choices": []}, status=500))
        with _creds():
            out = openrouter.generate_batch(["p"], model="m", session_fn=lambda: sess)
        self.assertFalse(out[0]["ok"])
        self.assertEqual(out[0]["status"], 500)

    def test_an_unparseable_body_does_not_raise(self):
        sess = _Session({}, default=_Resp(_BAD, status=200, raw="<html>"))
        with _creds():
            out = openrouter.generate_batch(["p"], model="m", session_fn=lambda: sess)
        self.assertFalse(out[0]["ok"])

    def test_a_transport_exception_does_not_raise(self):
        sess = _Session({}, default=RuntimeError("connection reset"))
        with _creds():
            out = openrouter.generate_batch(["p"], model="m", session_fn=lambda: sess)
        self.assertEqual(out, [{"text": "", "ok": False}])


class TestJsonExtraction(unittest.TestCase):
    def test_a_bare_object_passes_through(self):
        self.assertEqual(openrouter._extract_json('{"tags": ["a"]}'), '{"tags": ["a"]}')

    def test_an_object_after_reasoning_prose_is_found(self):
        text = 'We need to consider the vocabulary.\n\nSo:\n{"tags": ["mqtt"]}'
        self.assertEqual(json.loads(openrouter._extract_json(text)), {"tags": ["mqtt"]})

    def test_nested_objects_are_balanced_correctly(self):
        text = 'thinking... {"a": {"b": 1}, "c": 2} done'
        self.assertEqual(json.loads(openrouter._extract_json(text)), {"a": {"b": 1}, "c": 2})

    def test_prose_with_no_object_returns_none(self):
        self.assertIsNone(openrouter._extract_json("I cannot answer that."))
        self.assertIsNone(openrouter._extract_json(""))

    def test_a_brace_that_is_not_json_is_not_mistaken_for_one(self):
        self.assertIsNone(openrouter._extract_json("the set {a, b} of tags"))

    def test_fmt_json_rejects_a_response_with_no_object(self):
        sess = _Session({}, default=_Resp(_ok("no json here")))
        with _creds():
            out = openrouter.generate_batch(["p"], model="m", fmt="json",
                                            session_fn=lambda: sess)
        self.assertFalse(out[0]["ok"])

    def test_fmt_json_accepts_an_object_buried_in_prose(self):
        sess = _Session({}, default=_Resp(_ok('sure!\n{"tags": ["mqtt"]}')))
        with _creds():
            out = openrouter.generate_batch(["p"], model="m", fmt="json",
                                            session_fn=lambda: sess)
        self.assertTrue(out[0]["ok"])
        self.assertEqual(json.loads(out[0]["text"]), {"tags": ["mqtt"]})


class TestLadder(unittest.TestCase):
    def test_it_falls_through_to_the_next_model_on_a_rate_limit(self):
        sess = _Session({
            "a": _Resp({"error": {"code": 429, "message": "rate limited"}}),
            "b": _Resp(_ok("answer")),
        })
        with _creds():
            out = openrouter.generate_batch(["p"], ladder=("a", "b"),
                                            session_fn=lambda: sess)
        self.assertTrue(out[0]["ok"])
        self.assertEqual(out[0]["text"], "answer")
        self.assertEqual(sess.calls, ["a", "b"])

    def test_it_stops_as_soon_as_every_prompt_is_answered(self):
        sess = _Session({"a": _Resp(_ok("answer"))})
        with _creds():
            openrouter.generate_batch(["p"], ladder=("a", "b", "c"),
                                      session_fn=lambda: sess)
        self.assertEqual(sess.calls, ["a"], "b and c must not be called")

    def test_only_the_prompts_that_failed_are_retried(self):
        calls = []

        class S:
            def post(self, url, headers=None, json=None, timeout=None):
                model, prompt = json["model"], json["messages"][0]["content"]
                calls.append((model, prompt))
                if model == "a" and prompt == "p2":
                    return _Resp({"error": {"code": 429, "message": "x"}})
                return _Resp(_ok(f"{model}:{prompt}"))

        with _creds():
            out = openrouter.generate_batch(["p1", "p2"], ladder=("a", "b"),
                                            session_fn=lambda: S())
        self.assertEqual(out[0]["text"], "a:p1")
        self.assertEqual(out[1]["text"], "b:p2")
        self.assertEqual(sorted(calls), [("a", "p1"), ("a", "p2"), ("b", "p2")])

    def test_the_result_list_always_aligns_with_the_prompts(self):
        sess = _Session({}, default=_Resp({"error": {"code": 429, "message": "x"}}))
        with _creds():
            out = openrouter.generate_batch(["p1", "p2", "p3"], ladder=("a",),
                                            session_fn=lambda: sess)
        self.assertEqual(len(out), 3)
        self.assertTrue(all(r == {"text": "", "ok": False, "status": 429} for r in out))

    def test_the_default_ladder_tries_free_before_paid(self):
        self.assertEqual(openrouter.DEFAULT_LADDER[:len(openrouter.FREE_LADDER)],
                         openrouter.FREE_LADDER)
        self.assertTrue(set(openrouter.CHEAP_LADDER) <= set(openrouter.DEFAULT_LADDER))


class TestCredentials(unittest.TestCase):
    def test_no_key_returns_degraded_results_rather_than_raising(self):
        with mock.patch.object(openrouter, "credentials", lambda: ("https://x", "")):
            out = openrouter.generate_batch(["p1", "p2"])
        self.assertEqual(out, [{"text": "", "ok": False}] * 2)

    def test_available_reflects_the_key(self):
        with mock.patch.object(openrouter, "credentials", lambda: ("https://x", "")):
            self.assertFalse(openrouter.available())
        with mock.patch.object(openrouter, "credentials", lambda: ("https://x", "k")):
            self.assertTrue(openrouter.available())

    def test_an_empty_prompt_list_costs_nothing(self):
        self.assertEqual(openrouter.generate_batch([]), [])


if __name__ == "__main__":
    unittest.main()
