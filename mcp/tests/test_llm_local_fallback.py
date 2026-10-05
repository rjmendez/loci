"""llm_local: say why it failed, and try the tier that works.

generate() targeted Ollama only, with a hardcoded model tag, and swallowed the
exception on failure — returning {'text':'','ok':False} with nothing to act on.

Measured on this host: Ollama serves ONLY nomic-embed-text (an embedding model,
no generation model at all), so a hardcoded generation tag cannot work.
Asking either server for the other's name fails, and the reason was discarded.

Consequence: verify_finding returned degraded=True with empty reasoning for every
finding, and the verify groom pass refused to run (correctly) rather than write
uncertain/0.0 records. The cause sat in a swallowed exception.
"""
import os
import unittest
from unittest import mock

import llm_local


class FailuresAreExplainedTest(unittest.TestCase):

    def test_an_empty_prompt_says_so(self):
        r = llm_local.generate("")
        self.assertFalse(r["ok"])
        self.assertIn("empty prompt", r["why"])

    def test_no_endpoint_says_so(self):
        with mock.patch.object(llm_local, "_gen_env", lambda: ""), \
             mock.patch.object(llm_local, "_resolve_ollama", return_value=""):
            r = llm_local.generate("hello")
        self.assertFalse(r["ok"])
        self.assertIn("no Ollama endpoint", r["why"])

    def test_a_transport_failure_carries_the_exception(self):
        """The regression: this used to return ok=False with no reason at all."""
        with mock.patch.object(llm_local, "_gen_env", lambda: "http://x"), \
             mock.patch("requests.post", side_effect=OSError("connection refused")):
            r = llm_local.generate("hello")
        self.assertFalse(r["ok"])
        self.assertIn("OSError", r["why"])
        self.assertIn("connection refused", r["why"])

    def test_unparseable_json_says_so_rather_than_just_failing(self):
        resp = mock.MagicMock()
        resp.json.return_value = {"response": "not json at all"}
        with mock.patch.object(llm_local, "_gen_env", lambda: "http://x"), \
             mock.patch("requests.post", return_value=resp):
            r = llm_local.generate("hello", fmt="json")
        self.assertFalse(r["ok"])
        self.assertIn("not valid JSON", r["why"])


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
             mock.patch.object(llm_local, "_gen_env", lambda: "http://x"), \
             mock.patch("requests.post", return_value=resp) as post:
            r = llm_local.generate("hello")
        self.assertEqual(r["model"], "some-model:7b")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "some-model:7b")


class CloudTierFallbackTest(unittest.TestCase):
    def test_cloud_tier_openrouter_fallback_returns_success(self):
        with mock.patch.dict("os.environ", {"LOCI_CLOUD_TIER_ALLOW_PROMPT_EXPORT": "1"}), \
             mock.patch.object(llm_local, "_gen_env", lambda: "http://x"), \
             mock.patch("requests.post", side_effect=OSError("refused")), \
             mock.patch.object(llm_local, "_supervisor_route",
                               return_value={"provider": "openrouter", "role": "triage"}), \
             mock.patch("backends.cloud_tier_enabled", return_value=True), \
             mock.patch("backends.openrouter_model", return_value="qwen/qwen3.8-27b:free"), \
             mock.patch("backends.abliteration_model", return_value="abliterated-model"), \
             mock.patch("openrouter.generate_batch",
                        return_value=[{"text": "ok-from-openrouter", "ok": True, "model": "or"}]):
            r = llm_local.generate("hello")
        self.assertTrue(r["ok"])
        self.assertEqual(r["tier"], "cloud-openrouter")

    def test_cloud_tier_prefers_abliteration_for_redteam_route(self):
        with mock.patch.dict("os.environ", {"LOCI_CLOUD_TIER_ALLOW_PROMPT_EXPORT": "1"}), \
             mock.patch.object(llm_local, "_gen_env", lambda: "http://x"), \
             mock.patch("requests.post", side_effect=OSError("refused")), \
             mock.patch.object(llm_local, "_supervisor_route",
                               return_value={"provider": "abliteration", "role": "redteam"}), \
             mock.patch("backends.cloud_tier_enabled", return_value=True), \
             mock.patch("backends.openrouter_model", return_value="qwen/qwen3.8-flash"), \
             mock.patch("backends.abliteration_model", return_value="abliterated-model-large-v2"), \
             mock.patch("abliteration.generate_batch",
                        return_value=[{"text": "ok-from-abliteration", "ok": True, "model": "ab"}]):
            r = llm_local.generate("simulate adversarial exploit prompt")
        self.assertTrue(r["ok"])
        self.assertEqual(r["tier"], "cloud-abliteration")

    def test_cloud_tier_requires_explicit_prompt_export_opt_in(self):
        with mock.patch.dict("os.environ", {}, clear=False), \
             mock.patch.object(llm_local, "_gen_env", lambda: "http://x"), \
             mock.patch("requests.post", side_effect=OSError("refused")), \
             mock.patch.object(llm_local, "_supervisor_route",
                               return_value={"provider": "openrouter", "role": "triage"}), \
             mock.patch("backends.cloud_tier_enabled", return_value=True), \
             mock.patch("openrouter.generate_batch") as gen:
            import os
            os.environ.pop("LOCI_CLOUD_TIER_ALLOW_PROMPT_EXPORT", None)
            r = llm_local.generate("hello")
        self.assertFalse(r["ok"])
        gen.assert_not_called()

    def test_cloud_tier_refuses_when_per_call_token_cap_is_exceeded(self):
        with mock.patch.dict("os.environ", {
            "LOCI_CLOUD_TIER_ALLOW_PROMPT_EXPORT": "1",
            "LOCI_CLOUD_TIER_MAX_TOKENS_PER_CALL": "16",
        }), \
             mock.patch.object(llm_local, "_gen_env", lambda: "http://x"), \
             mock.patch("requests.post", side_effect=OSError("refused")), \
             mock.patch.object(llm_local, "_supervisor_route",
                               return_value={"provider": "openrouter", "role": "triage"}), \
             mock.patch("backends.cloud_tier_enabled", return_value=True), \
             mock.patch("backends.openrouter_model", return_value="qwen/qwen3.8-27b:free"), \
             mock.patch("backends.abliteration_model", return_value="abliterated-model"), \
             mock.patch("openrouter.generate_batch") as gen:
            r = llm_local.generate("hello", max_tokens=64)
        self.assertFalse(r["ok"])
        self.assertEqual(r.get("tier"), "cloud-refused")
        self.assertIn("per-call cap", r.get("why", ""))
        gen.assert_not_called()

    def test_cloud_tier_downgrades_when_preferred_provider_is_denied(self):
        with mock.patch.dict("os.environ", {
            "LOCI_CLOUD_TIER_ALLOW_PROMPT_EXPORT": "1",
            "LOCI_CLOUD_TIER_DENY_PROVIDERS": "abliteration",
        }), \
             mock.patch.object(llm_local, "_gen_env", lambda: "http://x"), \
             mock.patch("requests.post", side_effect=OSError("refused")), \
             mock.patch.object(llm_local, "_supervisor_route",
                               return_value={"provider": "abliteration", "role": "redteam"}), \
             mock.patch("backends.cloud_tier_enabled", return_value=True), \
             mock.patch("backends.openrouter_model", return_value="qwen/qwen3.8-flash"), \
             mock.patch("backends.abliteration_model", return_value="abliterated-model-large-v2"), \
             mock.patch("openrouter.generate_batch",
                        return_value=[{"text": "fallback-openrouter", "ok": True, "model": "or"}]) as or_gen, \
             mock.patch("abliteration.generate_batch") as ab_gen:
            r = llm_local.generate("simulate adversarial exploit prompt")
        self.assertTrue(r["ok"])
        self.assertEqual(r["tier"], "cloud-openrouter")
        or_gen.assert_called_once()
        ab_gen.assert_not_called()

    def test_cloud_tier_refuses_when_daily_call_budget_is_exhausted(self):
        budget_state = os.path.join(os.path.dirname(__file__), "_cloud_budget_state_test.json")
        if os.path.exists(budget_state):
            os.remove(budget_state)
        try:
            with mock.patch.dict("os.environ", {
                "LOCI_CLOUD_TIER_ALLOW_PROMPT_EXPORT": "1",
                "LOCI_CLOUD_TIER_DAILY_CALL_BUDGET": "1",
                "LOCI_CLOUD_TIER_BUDGET_STATE_PATH": budget_state,
            }), \
                 mock.patch.object(llm_local, "_gen_env", lambda: "http://x"), \
                 mock.patch("requests.post", side_effect=OSError("refused")), \
                 mock.patch.object(llm_local, "_supervisor_route",
                                   return_value={"provider": "openrouter", "role": "triage"}), \
                 mock.patch("backends.cloud_tier_enabled", return_value=True), \
                 mock.patch("backends.openrouter_model", return_value="qwen/qwen3.8-27b:free"), \
                 mock.patch("backends.abliteration_model", return_value="abliterated-model"), \
                 mock.patch("openrouter.generate_batch",
                            return_value=[{"text": "ok-from-openrouter", "ok": True, "model": "or"}]) as gen:
                first = llm_local.generate("hello-1")
                second = llm_local.generate("hello-2")
            self.assertTrue(first["ok"])
            self.assertFalse(second["ok"])
            self.assertEqual(second.get("tier"), "cloud-refused")
            self.assertIn("daily call budget exhausted", second.get("why", ""))
            self.assertEqual(gen.call_count, 1)
        finally:
            if os.path.exists(budget_state):
                os.remove(budget_state)


class GenerationEnvPrecedenceTest(unittest.TestCase):
    """OLLAMA_BASE_URL is the EMBEDDING endpoint and must not outrank generation.

    #222 taught the RESOLVER to separate the two, then left the embedding variable
    as the highest-precedence generation override. scripts/loci_groom.load_env(),
    which every groom pass calls, sets OLLAMA_BASE_URL to the in-cluster host that
    serves only nomic-embed-text — so under cron, 100% of generation went to a
    host with no generation model (a fallback tier that has since been removed had been
    silently rescuing it; without it the groom pass went from 9/100 to 100/100 degraded).

    Read at CALL time so load_env() — which runs after import — is respected, and
    so these tests need no importlib.reload (which made the first version of them
    order-dependent inside the full suite).
    """

    def test_the_embedding_var_does_not_capture_generation(self):
        with mock.patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://embeddings-only:11434"}):
            self.assertEqual(llm_local._gen_env(), "",
                             "OLLAMA_BASE_URL must not become the generation endpoint")

    def test_a_generation_specific_var_still_wins(self):
        for var in ("LOCI_OLLAMA_GEN_URL", "OLLAMA_GEN_URL"):
            with self.subTest(var=var):
                env = {k: v for k, v in os.environ.items()
                       if k not in ("LOCI_OLLAMA_GEN_URL", "OLLAMA_GEN_URL")}
                env[var] = "http://gen:1"
                with mock.patch.dict("os.environ", env, clear=True):
                    self.assertEqual(llm_local._gen_env(), "http://gen:1")

    def test_env_is_read_at_call_time_not_import_time(self):
        """load_env() runs AFTER this module is imported; an import-time capture
        would reflect the environment before the process configured itself."""
        with mock.patch.dict("os.environ", {"LOCI_OLLAMA_GEN_URL": "http://late:2"}):
            self.assertEqual(llm_local._gen_env(), "http://late:2")
        self.assertNotEqual(llm_local._gen_env(), "http://late:2")

    def test_a_single_capable_ollama_still_resolves_via_backends(self):
        """No regression for a host that only sets OLLAMA_BASE_URL: the resolver
        falls through ollama_gen_url() -> ollama_url()."""
        import backends
        with mock.patch.object(backends, "ollama_gen_url", return_value="http://only:11434"):
            self.assertEqual(llm_local._resolve_ollama(), "http://only:11434")
