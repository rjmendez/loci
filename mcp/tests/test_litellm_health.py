"""litellm_health / check_proxy_health against a real local HTTP server (no stubbed network).

On the live machine the proxy answered 401 and the tool reported "unreachable": HTTPError is a
URLError, so the auth failure was filed as no answer at all.

Mutation log:
- move the URLError branch ahead of HTTPError: test_401_is_auth_required_not_unreachable red
- drop the Authorization header: test_key_is_sent_and_accepted red
- report every HTTPError as ok: test_500_is_an_error_that_was_reached red
- probe /health instead of /health/liveliness + /v1/models: test_200_is_ok, test_key_is_sent_and_accepted,
  test_the_expensive_health_endpoint_is_never_requested red (on the live machine /health with a key checks every model
  deployment and timed out, so a healthy proxy read "unreachable")
- skip the liveliness probe: test_a_down_liveliness_probe_is_an_error_and_models_is_not_asked red
- send the key to the liveliness probe: test_200_is_ok, test_key_is_sent_and_accepted red
"""
from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from loci import litellm_client  # noqa: E402
from loci.litellm_client import check_proxy_health  # noqa: E402


class _Proxy:
    def __init__(self, code=200, require_key=None, live_code=200):
        outer = self
        self.code, self.require_key, self.live_code, self.seen = code, require_key, live_code, []

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.seen.append((self.path, self.headers.get("Authorization")))
                if self.path == "/health/liveliness":
                    self.send_response(outer.live_code)
                elif self.path != "/v1/models":
                    self.send_response(404)             # anything else, /health included, is not what the probe may ask
                elif outer.require_key and self.headers.get("Authorization") != f"Bearer {outer.require_key}":
                    self.send_response(401)
                else:
                    self.send_response(outer.code)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def proxy():
    made = []

    def make(**kw):
        p = _Proxy(**kw)
        made.append(p)
        return p
    yield make
    for p in made:
        p.close()


def test_200_is_ok(proxy):
    p = proxy(code=200)
    assert check_proxy_health(p.url, api_key="") == {
        "status": "ok", "reachable": True, "proxy_url": p.url, "http_status": 200, "checked": "/v1/models"}
    assert p.seen == [("/health/liveliness", None), ("/v1/models", None)]


def test_401_is_auth_required_not_unreachable(proxy):
    p = proxy(require_key="secret")
    r = check_proxy_health(p.url, api_key="")
    assert r["status"] == "auth_required"
    assert r["reachable"] is True
    assert r["http_status"] == 401
    assert r["key_sent"] is False
    assert r["hint"] == "set LOCI_LITELLM_API_KEY to a key the proxy accepts"


def test_key_is_sent_and_accepted(proxy):
    p = proxy(require_key="secret")
    r = check_proxy_health(p.url, api_key="secret")
    assert (r["status"], r["http_status"]) == ("ok", 200)
    assert p.seen == [("/health/liveliness", None), ("/v1/models", "Bearer secret")]


def test_a_wrong_key_is_auth_required_and_says_a_key_was_sent(proxy):
    p = proxy(require_key="secret")
    r = check_proxy_health(p.url, api_key="wrong")
    assert (r["status"], r["key_sent"]) == ("auth_required", True)


def test_500_is_an_error_that_was_reached(proxy):
    p = proxy(code=500)
    r = check_proxy_health(p.url, api_key="")
    assert (r["status"], r["reachable"], r["http_status"], r["checked"]) == ("error", True, 500, "/v1/models")


def test_a_down_liveliness_probe_is_an_error_and_models_is_not_asked(proxy):
    p = proxy(live_code=503)
    r = check_proxy_health(p.url, api_key="secret")
    assert (r["status"], r["reachable"], r["http_status"], r["checked"]) == ("error", True, 503, "/health/liveliness")
    assert p.seen == [("/health/liveliness", None)]


def test_a_401_on_liveliness_is_an_error_not_a_key_problem(proxy):
    p = proxy(live_code=401)
    r = check_proxy_health(p.url, api_key="secret")
    assert (r["status"], r["http_status"], r["checked"]) == ("error", 401, "/health/liveliness")
    assert "key_sent" not in r


def test_the_expensive_health_endpoint_is_never_requested(proxy):
    p = proxy(require_key="secret")
    check_proxy_health(p.url, api_key="secret")
    check_proxy_health(p.url, api_key="")
    assert "/health" not in [path for path, _ in p.seen]
    assert [path for path, _ in p.seen] == ["/health/liveliness", "/v1/models"] * 2


def test_nothing_listening_is_unreachable():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    r = check_proxy_health(f"http://127.0.0.1:{port}", api_key="", timeout=2)
    assert (r["status"], r["reachable"]) == ("unreachable", False)
    assert r["proxy_url"] == f"http://127.0.0.1:{port}"


def test_trailing_slash_in_the_url_is_tolerated(proxy):
    p = proxy(code=200)
    assert check_proxy_health(p.url + "/", api_key="")["proxy_url"] == p.url


def test_tool_reports_the_same_result_as_the_probe(proxy, monkeypatch):
    sys.path.insert(0, str(_REPO / "mcp"))
    from tools import litellm_tools

    class FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self):
            return lambda fn: self.tools.setdefault(fn.__name__, fn)

    fake = FakeServer()
    litellm_tools.register(fake)
    p = proxy(require_key="secret")
    monkeypatch.setattr(litellm_client, "LITELLM_PROXY_URL", p.url)
    monkeypatch.setattr(litellm_client, "LITELLM_API_KEY", "")
    out = json.loads(fake.tools["litellm_health"]())
    assert (out["status"], out["reachable"], out["http_status"]) == ("auth_required", True, 401)
    monkeypatch.setattr(litellm_client, "LITELLM_API_KEY", "secret")
    out = json.loads(fake.tools["litellm_health"]())
    assert (out["status"], out["http_status"], out["checked"]) == ("ok", 200, "/v1/models")
