"""ollama_lab_relay: header rewrite, body passthrough, incremental streaming, error shape.

A fake upstream HTTP server stands in for the Ollama lab name; the relay under test is
pointed at it by replacing the module's UPSTREAM, so nothing touches the network.
"""
import http.client
import importlib.util
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "ollama_lab_relay", Path(__file__).resolve().parents[1] / "ollama_lab_relay.py")
relay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(relay)


class FakeUpstream(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def _record(self, body=b""):
        FakeUpstream.seen.append({"method": self.command, "path": self.path,
                                  "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})

    def do_GET(self):
        self._record()
        if self.path == "/missing":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"nope")
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for i in range(3):
                self.wfile.write(json.dumps({"n": i}).encode() + b"\n")
                self.wfile.flush()
                time.sleep(0.4)
        else:
            body = b'{"models":[]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Resident-Model", "resident")
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._record(body)
        out = json.dumps({"echo": json.loads(body or b"{}")}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_DELETE(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._record(body)
        self.send_response(204)
        self.end_headers()


def serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def stack(monkeypatch):
    FakeUpstream.seen.clear()
    up = serve(FakeUpstream)
    monkeypatch.setattr(relay, "UPSTREAM", urllib.parse.urlsplit(f"http://127.0.0.1:{up.server_port}"))
    front = serve(relay.Relay)
    yield front.server_port, up.server_port
    front.shutdown()
    up.shutdown()


def call(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    return r.status, dict(r.getheaders()), r.read()


def test_get_passthrough_and_host_rewrite(stack):
    front, up = stack
    status, headers, body = call(front, "GET", "/api/tags", headers={"Host": "127.0.0.1:11434", "X-Test": "1"})
    assert status == 200 and json.loads(body) == {"models": []}
    assert headers.get("X-Resident-Model") == "resident"
    seen = FakeUpstream.seen[-1]
    assert seen["headers"]["host"] == f"127.0.0.1:{up}"          # the upstream's name, not the caller's
    assert seen["headers"]["x-test"] == "1"


def test_post_body_roundtrip(stack):
    front, _ = stack
    payload = json.dumps({"model": "nomic-embed-text", "input": "hello"})
    status, _, body = call(front, "POST", "/v1/embeddings", payload, {"Content-Type": "application/json"})
    assert status == 200 and json.loads(body)["echo"]["input"] == "hello"
    assert json.loads(FakeUpstream.seen[-1]["body"])["model"] == "nomic-embed-text"


def test_upstream_status_passes_through(stack):
    front, _ = stack
    status, _, body = call(front, "GET", "/missing")
    assert status == 404 and body == b"nope"


def test_response_streams_incrementally(stack):
    """The first line must reach the client well before the upstream finishes (about 1.2 s)."""
    front, _ = stack
    c = http.client.HTTPConnection("127.0.0.1", front, timeout=10)
    t0 = time.monotonic()
    c.request("GET", "/stream")
    r = c.getresponse()
    first = r.readline()
    first_at = time.monotonic() - t0
    rest = r.read()
    assert json.loads(first) == {"n": 0}
    assert first_at < 0.9, f"first chunk took {first_at:.2f}s: relay is buffering the stream"
    assert rest.count(b"\n") == 2


def test_unreachable_upstream_is_a_502_with_json(monkeypatch):
    monkeypatch.setattr(relay, "UPSTREAM", urllib.parse.urlsplit("http://127.0.0.1:9"))   # discard port, nothing listens
    front = serve(relay.Relay)
    try:
        status, headers, body = call(front.server_port, "GET", "/api/tags")
    finally:
        front.shutdown()
    assert status == 502
    assert "relay upstream" in json.loads(body)["error"]


def test_hop_by_hop_headers_are_not_forwarded(stack):
    front, _ = stack
    call(front, "GET", "/api/tags", headers={"Connection": "keep-alive", "Keep-Alive": "timeout=5", "TE": "trailers"})
    seen = FakeUpstream.seen[-1]["headers"]
    assert "keep-alive" not in seen and "te" not in seen


def test_default_upstream_is_the_lab_name_not_an_ip():
    spec = importlib.util.spec_from_file_location(
        "ollama_lab_relay_default", Path(__file__).resolve().parents[1] / "ollama_lab_relay.py")
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    host = fresh.UPSTREAM.hostname
    assert fresh.UPSTREAM.scheme == "https" and host.endswith(".lab.floppydicks.net")
    assert not all(part.isdigit() for part in host.split("."))


def test_delete_with_body_is_forwarded_with_its_method(stack):
    """Ollama's model delete is DELETE /api/delete with a JSON body; the relay serves every method, not just GET and POST."""
    front, _ = stack
    conn = http.client.HTTPConnection("127.0.0.1", front, timeout=10)
    conn.request("DELETE", "/api/delete", body=b'{"model":"m:1"}', headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 204
    assert [(r["method"], r["path"], r["body"]) for r in FakeUpstream.seen] == [("DELETE", "/api/delete", b'{"model":"m:1"}')]
