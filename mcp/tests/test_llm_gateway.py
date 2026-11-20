"""Loci's generation and embedding calls go through an OpenAI-compatible gateway when one is configured,
and never reach a raw Ollama endpoint behind the operator's back."""
import http.server
import json
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import backends  # noqa: E402
import embed_ops  # noqa: E402
import llm_gateway as G  # noqa: E402
import llm_local  # noqa: E402
import pytest  # noqa: E402
from memcheck import llm as memllm  # noqa: E402


class _FakeGateway:
    """A real HTTP server speaking the two OpenAI endpoints the client uses."""

    def __init__(self):
        self.requests: list[dict] = []
        self.chat_status = 200
        self.chat_content = "hello from the gateway"
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                if self.path.endswith("/chat/completions"):
                    status, payload = outer.chat_status, {
                        "model": "served-by-gateway",
                        "choices": [{"message": {"role": "assistant", "content": outer.chat_content}}]}
                elif self.path.endswith("/embeddings"):
                    inputs = body["input"]
                    status, payload = 200, {"data": [{"index": i, "embedding": [float(i), 1.0]}
                                                      for i in reversed(range(len(inputs)))]}
                else:
                    status, payload = 404, {"error": "no such path"}
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def gateway(monkeypatch, tmp_path):
    server = _FakeGateway()
    for var in ("LOCI_LLM_GATEWAY_URL", "LOCI_LLM_GATEWAY_KEY", "LOCI_LLM_GATEWAY_MODEL",
                "LOCI_LLM_GATEWAY_EMBED_MODEL", "LOCI_LLM_GATEWAY_FALLBACK_RAW", "LOCI_LLM_GATEWAY_TIMEOUT",
                "MEMCHECK_LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(backends, "_CONFIG_PATH", str(tmp_path / "missing.toml"))
    backends._reset_cache()
    yield server
    server.close()
    backends._reset_cache()


def _configure(monkeypatch, server, **extra):
    monkeypatch.setenv("LOCI_LLM_GATEWAY_URL", server.url)
    monkeypatch.setenv("LOCI_LLM_GATEWAY_KEY", "sk-test")
    monkeypatch.setenv("LOCI_LLM_GATEWAY_MODEL", "chat-group")
    monkeypatch.setenv("LOCI_LLM_GATEWAY_EMBED_MODEL", "embed-group")
    for k, v in extra.items():
        monkeypatch.setenv(k, v)


def test_it_is_off_without_a_url(gateway):
    assert not G.enabled()
    assert G.chat("hi")["ok"] is False
    assert G.embed(["x"]) == []


def test_chat_posts_to_v1_with_key_model_and_the_internal_flag(gateway, monkeypatch):
    _configure(monkeypatch, gateway)

    out = G.chat("say hi", max_tokens=12, json_mode=True)

    assert out["ok"] and out["text"] == "hello from the gateway" and out["model"] == "served-by-gateway"
    req = gateway.requests[0]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer sk-test"
    assert req["body"]["model"] == "chat-group"
    assert req["body"]["messages"] == [{"role": "user", "content": "say hi"}]
    assert req["body"]["response_format"] == {"type": "json_object"}
    assert req["body"]["metadata"] == {G.INTERNAL_CALL_FLAG: True}


def test_a_configured_gateway_without_a_model_group_fails_loudly(gateway, monkeypatch):
    monkeypatch.setenv("LOCI_LLM_GATEWAY_URL", gateway.url)

    out = G.chat("hi")

    assert out["ok"] is False and "no model group" in out["why"]
    assert gateway.requests == []


def test_http_errors_are_reported_not_raised(gateway, monkeypatch):
    _configure(monkeypatch, gateway)
    gateway.chat_status = 429

    out = G.chat("hi")

    assert out["ok"] is False and "429" in out["why"]


def test_the_private_config_supplies_the_settings(gateway, monkeypatch, tmp_path):
    cfg = tmp_path / "backends.toml"
    cfg.write_text(f'[llm_gateway]\nurl = "{gateway.url}/v1"\nkey = "cfg-key"\nmodel = "cfg-group"\n')
    monkeypatch.setattr(backends, "_CONFIG_PATH", str(cfg))
    backends._reset_cache()

    assert G.chat("hi")["ok"]
    assert gateway.requests[0]["auth"] == "Bearer cfg-key" and gateway.requests[0]["body"]["model"] == "cfg-group"


def test_embeddings_come_back_in_input_order(gateway, monkeypatch):
    _configure(monkeypatch, gateway)

    assert G.embed(["a", "b", "c"]) == [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    assert gateway.requests[0]["body"]["model"] == "embed-group"
    assert gateway.requests[0]["body"]["metadata"] == {G.INTERNAL_CALL_FLAG: True}


def test_llm_local_generate_goes_through_the_gateway_and_never_touches_ollama(gateway, monkeypatch):
    _configure(monkeypatch, gateway)
    raw_calls = []
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: raw_calls.append(a) or (_ for _ in ()).throw(AssertionError))

    out = llm_local._generate("write one short sentence", model="some-local:4b", max_tokens=20)

    assert out["ok"] and out["text"] == "hello from the gateway" and out["tier"] == "gateway"
    assert raw_calls == []


def test_llm_local_json_mode_still_validates_the_body(gateway, monkeypatch):
    _configure(monkeypatch, gateway)
    gateway.chat_content = "not json"

    out = llm_local._generate("give me json", fmt="json")

    assert out["ok"] is False and "not valid JSON" in out["why"]


def test_a_gateway_failure_does_not_silently_fall_back_to_raw_ollama(gateway, monkeypatch):
    _configure(monkeypatch, gateway)
    gateway.chat_status = 500
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("raw call")))

    out = llm_local._generate("write one short sentence")

    assert out["ok"] is False and "500" in out["why"]


def test_embed_ops_uses_the_gateway(gateway, monkeypatch):
    _configure(monkeypatch, gateway)

    assert embed_ops.embed_texts(["a", "b"]) == [[0.0, 1.0], [1.0, 1.0]]
    assert gateway.requests[0]["path"] == "/v1/embeddings"


def test_memcheck_prefers_the_gateway_for_chat_and_embeddings(gateway, monkeypatch):
    _configure(monkeypatch, gateway)

    assert memllm.llm_available()
    assert memllm._provider_order() == ["gateway"]
    assert memllm.call_llm("judge this", max_tokens=8) == "hello from the gateway"
    assert memllm.embed_texts(["a", "b"]) == [[0.0, 1.0], [1.0, 1.0]]


def test_memcheck_keeps_the_raw_chain_behind_the_explicit_fallback_switch(gateway, monkeypatch):
    _configure(monkeypatch, gateway, LOCI_LLM_GATEWAY_FALLBACK_RAW="1")

    assert memllm._provider_order()[:2] == ["gateway", "ollama"]
