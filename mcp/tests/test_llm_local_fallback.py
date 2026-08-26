"""llm_local: say why it failed, and try the tier that works.

generate() targeted Ollama only, with a hardcoded model tag, and swallowed the
exception on failure — returning {'text':'','ok':False} with nothing to act on.

Measured on this host: Ollama serves ONLY nomic-embed-text (an embedding model,
no generation model at all) while vLLM serves the generation model under a
different name — Qwen/Qwen2.5-3B-Instruct, not the Ollama-style qwen2.5:3b.
Asking either server for the other's name fails, and the reason was discarded.

Consequence: verify_finding returned degraded=True with empty reasoning for every
finding, and the verify groom pass refused to run (correctly) rather than write
uncertain/0.0 records. The cause sat in a swallowed exception.
"""
import unittest
from unittest import mock

import llm_local


class FailuresAreExplainedTest(unittest.TestCase):

    def test_an_empty_prompt_says_so(self):
        r = llm_local.generate("")
        self.assertFalse(r["ok"])
        self.assertIn("empty prompt", r["why"])

    def test_no_endpoint_says_so(self):
        with mock.patch.object(llm_local, "_OLLAMA", ""), \
             mock.patch.object(llm_local, "_resolve_ollama", return_value=""):
            r = llm_local.generate("hello")
        self.assertFalse(r["ok"])
        self.assertIn("no Ollama endpoint", r["why"])

    def test_a_transport_failure_carries_the_exception(self):
        """The regression: this used to return ok=False with no reason at all."""
        with mock.patch.object(llm_local, "_OLLAMA", "http://x"), \
             mock.patch.object(llm_local, "_try_vllm", return_value=None), \
             mock.patch("requests.post", side_effect=OSError("connection refused")):
            r = llm_local.generate("hello")
        self.assertFalse(r["ok"])
        self.assertIn("OSError", r["why"])
        self.assertIn("connection refused", r["why"])

    def test_unparseable_json_says_so_rather_than_just_failing(self):
        resp = mock.MagicMock()
        resp.json.return_value = {"response": "not json at all"}
        with mock.patch.object(llm_local, "_OLLAMA", "http://x"), \
             mock.patch("requests.post", return_value=resp):
            r = llm_local.generate("hello", fmt="json")
        self.assertFalse(r["ok"])
        self.assertIn("not valid JSON", r["why"])


class VllmFallbackTest(unittest.TestCase):

    def test_ollama_failure_falls_through_to_vllm(self):
        with mock.patch.object(llm_local, "_OLLAMA", "http://x"), \
             mock.patch("requests.post", side_effect=OSError("refused")), \
             mock.patch.object(llm_local, "_try_vllm",
                               return_value={"text": "OK", "ok": True,
                                             "model": "Qwen/Qwen2.5-3B-Instruct",
                                             "tier": "vllm"}):
            r = llm_local.generate("hello")
        self.assertTrue(r["ok"])
        self.assertEqual(r["tier"], "vllm")

    def test_a_successful_ollama_call_does_not_reach_vllm(self):
        resp = mock.MagicMock()
        resp.json.return_value = {"response": "hi"}
        with mock.patch.object(llm_local, "_OLLAMA", "http://x"), \
             mock.patch("requests.post", return_value=resp), \
             mock.patch.object(llm_local, "_try_vllm") as vllm:
            r = llm_local.generate("hello")
        self.assertTrue(r["ok"])
        vllm.assert_not_called()

    def test_vllm_declining_leaves_the_ollama_reason_intact(self):
        """Both tiers down must not hide WHICH failed or why."""
        with mock.patch.object(llm_local, "_OLLAMA", "http://x"), \
             mock.patch("requests.post", side_effect=OSError("refused")), \
             mock.patch.object(llm_local, "_try_vllm", return_value=None):
            r = llm_local.generate("hello")
        self.assertFalse(r["ok"])
        self.assertIn("ollama", r["why"])

    def test_a_vllm_result_that_is_not_ok_is_not_returned_as_success(self):
        with mock.patch.object(llm_local, "batched_gen", create=True):
            with mock.patch.dict("sys.modules"):
                fake = mock.MagicMock()
                fake.generate_batch.return_value = [{"text": "", "ok": False}]
                with mock.patch.dict("sys.modules", {"batched_gen": fake}):
                    self.assertIsNone(
                        llm_local._try_vllm("hi", fmt=None, max_tokens=8, temperature=0.0))


if __name__ == "__main__":
    unittest.main()


class GenerationEndpointResolutionTest(unittest.TestCase):
    """Reachability is not capability.

    backends.ollama_url() resolves env -> reachability probe -> config, and the
    probe only asks whether something ANSWERS. The in-cluster Ollama here serves
    only nomic-embed-text, so it passed the probe and then failed every
    generate() call. Measured: embeddings 93ms in-cluster vs 5595ms over the
    tailnet; the generation model exists only over the tailnet. One URL cannot
    serve both, so generation resolves separately.
    """

    def test_gen_url_prefers_its_own_env_var(self):
        import backends
        with mock.patch.dict("os.environ", {"LOCI_OLLAMA_GEN_URL": "http://gen:1"}):
            self.assertEqual(backends.ollama_gen_url(), "http://gen:1")

    def test_gen_url_prefers_config_over_the_embedding_url(self):
        import backends
        with mock.patch.dict("os.environ", {}, clear=False):
            os_env = {k: v for k, v in __import__("os").environ.items()
                      if k not in ("LOCI_OLLAMA_GEN_URL", "OLLAMA_GEN_URL")}
            with mock.patch.dict("os.environ", os_env, clear=True), \
                 mock.patch.object(backends, "_cfg", side_effect=lambda s, k, d="": {
                     ("ollama", "gen_url"): "http://oxalis:11434"}.get((s, k), d)), \
                 mock.patch.object(backends, "ollama_url", return_value="http://incluster:11434"):
                self.assertEqual(backends.ollama_gen_url(), "http://oxalis:11434")

    def test_gen_url_falls_back_to_the_embedding_url_when_unconfigured(self):
        """A host with one capable Ollama must need no extra config."""
        import backends
        with mock.patch.dict("os.environ", {}, clear=True), \
             mock.patch.object(backends, "_cfg", return_value=""), \
             mock.patch.object(backends, "ollama_url", return_value="http://only:11434"):
            self.assertEqual(backends.ollama_gen_url(), "http://only:11434")

    def test_generate_uses_the_configured_model_not_a_hardcoded_tag(self):
        import backends
        resp = mock.MagicMock()
        resp.json.return_value = {"response": "hi"}
        with mock.patch.object(backends, "ollama_gen_model", return_value="some-model:7b"), \
             mock.patch.object(llm_local, "_OLLAMA", "http://x"), \
             mock.patch("requests.post", return_value=resp) as post:
            r = llm_local.generate("hello")
        self.assertEqual(r["model"], "some-model:7b")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "some-model:7b")


class VllmFallbackIsOptInTest(unittest.TestCase):
    """The fallback must not borrow a service Loci does not own.

    backends resolves vLLM to 127.0.0.1:18000, which is
    /home/rjmendez/dama-vllm/vllm_tailscale_forward.py — another project's
    process, serving Qwen2.5-3B-Instruct at max_model_len=4096. Grounded verify
    prompts exceed that, so firing into it both 400s and consumes capacity Loci
    has no claim on. It was added when Ollama generation was broken; Ollama works
    now, so the default is off.
    """

    def test_disabled_by_default(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("LOCI_VLLM_FALLBACK", None)
            self.assertIsNone(
                llm_local._try_vllm("hi", fmt=None, max_tokens=8, temperature=0.0))

    def test_zero_is_also_disabled(self):
        with mock.patch.dict("os.environ", {"LOCI_VLLM_FALLBACK": "0"}):
            self.assertIsNone(
                llm_local._try_vllm("hi", fmt=None, max_tokens=8, temperature=0.0))

    def test_opt_in_reaches_batched_gen(self):
        fake = mock.MagicMock()
        fake.generate_batch.return_value = [{"text": "OK", "ok": True}]
        with mock.patch.dict("os.environ", {"LOCI_VLLM_FALLBACK": "1"}), \
             mock.patch.dict("sys.modules", {"batched_gen": fake}):
            out = llm_local._try_vllm("hi", fmt=None, max_tokens=8, temperature=0.0)
        self.assertEqual(out["ok"], True)
        fake.generate_batch.assert_called_once()

    def test_an_ollama_failure_no_longer_silently_reaches_vllm(self):
        """The whole point: a failed generate must report why, not reroute."""
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("LOCI_VLLM_FALLBACK", None)
            with mock.patch.object(llm_local, "_OLLAMA", "http://x"), \
                 mock.patch("requests.post", side_effect=OSError("refused")):
                r = llm_local.generate("hello")
        self.assertFalse(r["ok"])
        self.assertIn("ollama", r["why"])
