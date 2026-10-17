"""Per-call model/options, and a truncated prompt that says so.

Two defects an adversarial review of the tier census surfaced:

1. call_llm had no model parameter. The model resolved from _llm_model(), a
   process-global, so all four memcheck consumers shared one — making any
   per-consumer routing unexpressible in the code it targeted. _call_ollama also
   sent no options dict, so num_ctx and num_predict were unsettable.

2. Ollama silently truncates a prompt that exceeds the LOADED context window and
   returns done_reason="stop" with no error. Observed: prompt_eval_count=4095
   against a 4096 window, and a confidently wrong answer about content that had
   been cut. The same model has been seen loaded at 4096 and 16384 while
   DECLARING 131072 — the window is set by whoever loaded it, on a shared
   endpoint, so this reports rather than pinning num_ctx and forcing a reload
   that would evict another tenant.
"""
import unittest
from unittest import mock

from memcheck import llm


class PerCallModelAndOptionsTest(unittest.TestCase):

    def _capture(self, **kw):
        seen = {}

        def fake_post(url, payload, headers, timeout):
            seen.update(payload)
            return {"response": "ok", "prompt_eval_count": 999999}

        with mock.patch.object(llm, "_post_json", side_effect=fake_post), \
             mock.patch.object(llm, "_provider_order", return_value=["ollama"]):
            llm.call_llm("hello", **kw)
        return seen

    def test_model_defaults_to_the_configured_one(self):
        with mock.patch.object(llm, "_llm_model", return_value="configured:7b"):
            self.assertEqual(self._capture()["model"], "configured:7b")

    def test_an_explicit_model_overrides_the_global(self):
        """The whole point: two consumers in one process can differ."""
        with mock.patch.object(llm, "_llm_model", return_value="configured:7b"):
            self.assertEqual(self._capture(model="other:3b")["model"], "other:3b")

    def test_options_reach_the_payload(self):
        seen = self._capture(options={"num_ctx": 16384})
        self.assertEqual(seen["options"], {"num_ctx": 16384})

    def test_no_options_means_no_options_key(self):
        """Sending an empty options dict would still force server-side defaults."""
        self.assertNotIn("options", self._capture())

    def test_qwen_thinking_is_disabled_for_an_explicit_qwen_model(self):
        with mock.patch.object(llm, "_llm_model", return_value="llama:8b"):
            self.assertIs(self._capture(model="qwen3:14b")["think"], False)


class TruncationDetectionTest(unittest.TestCase):

    def _warn(self, prompt, evaluated):
        with mock.patch.object(llm, "_log") as log:
            llm._warn_if_truncated(prompt, {"prompt_eval_count": evaluated}, "m")
        return [c for c in log.warning.call_args_list]

    def test_a_truncated_prompt_warns(self):
        """The measured case: ~6000 tokens sent, 4095 evaluated."""
        warns = self._warn("x" * 24000, 4095)
        self.assertEqual(len(warns), 1)
        self.assertIn("TRUNCATED", warns[0].args[0])

    def test_a_normal_prompt_does_not_cry_wolf(self):
        """~4 chars/token is crude; the margin must tolerate ordinary variance."""
        self.assertEqual(self._warn("x" * 4000, 950), [])

    def test_a_slightly_short_count_is_tolerated(self):
        """Tokenisers legitimately beat 4 chars/token on repetitive text."""
        self.assertEqual(self._warn("x" * 4000, 700), [])

    def test_a_missing_count_is_not_a_warning(self):
        with mock.patch.object(llm, "_log") as log:
            llm._warn_if_truncated("x" * 24000, {}, "m")
        log.warning.assert_not_called()

    def test_a_garbage_count_does_not_raise(self):
        with mock.patch.object(llm, "_log") as log:
            llm._warn_if_truncated("x" * 100, {"prompt_eval_count": "many"}, "m")
        log.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class TruncationSignatureEdgesTest(unittest.TestCase):
    """The detector keys on 'stopped exactly at a power-of-two window'."""

    def _warn(self, prompt_chars, evaluated):
        with mock.patch.object(llm, "_log") as log:
            llm._warn_if_truncated("x" * prompt_chars,
                                   {"prompt_eval_count": evaluated}, "m")
        return log.warning.call_args_list

    def test_a_short_prompt_that_merely_lands_near_a_window_does_not_warn(self):
        """4096 tokens evaluated from a 4096-token prompt is a full read, not a cut."""
        self.assertEqual(self._warn(16000, 4096), [])

    def test_every_common_window_is_recognised(self):
        for w in (2048, 4096, 8192, 16384):
            with self.subTest(window=w):
                self.assertEqual(len(self._warn(w * 8, w - 1)), 1)

    def test_a_count_far_from_any_window_does_not_warn(self):
        self.assertEqual(self._warn(40000, 5000), [])
