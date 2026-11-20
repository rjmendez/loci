#!/usr/bin/env python3
"""Loopback shim: plain HTTP on 127.0.0.1:11434 -> the Ollama lab name over TLS.

Replaces the old raw-TCP forwarder that pointed at a tailnet IP and port that no
longer accepts connections. Everything that still says http://127.0.0.1:11434
now reaches Ollama through the name and the certificate, the same path the
migrated consumers use.

Env: OLLAMA_UPSTREAM_URL   default https://ollama.lab.floppydicks.net:4443
     OLLAMA_RELAY_BIND_HOST default 127.0.0.1
     OLLAMA_RELAY_BIND_PORT default 11434
     OLLAMA_RELAY_TIMEOUT   default 600 (seconds, per upstream read)

Deploy: copy to ~/ollama_relay_daemon.py (the cron @reboot line and the watchdog
use that path). See docs/OLLAMA_LAB_NAMES.md.
"""
import http.client
import os
import ssl
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = urllib.parse.urlsplit(os.environ.get("OLLAMA_UPSTREAM_URL", "https://ollama.lab.floppydicks.net:4443"))
BIND_HOST = os.environ.get("OLLAMA_RELAY_BIND_HOST", "127.0.0.1")
BIND_PORT = int(os.environ.get("OLLAMA_RELAY_BIND_PORT", "11434"))
TIMEOUT = float(os.environ.get("OLLAMA_RELAY_TIMEOUT", "600"))
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "host", "proxy-connection"}


def upstream_conn():
    port = UPSTREAM.port or (443 if UPSTREAM.scheme == "https" else 80)
    if UPSTREAM.scheme == "https":
        return http.client.HTTPSConnection(UPSTREAM.hostname, port, timeout=TIMEOUT,
                                           context=ssl.create_default_context())
    return http.client.HTTPConnection(UPSTREAM.hostname, port, timeout=TIMEOUT)


class Relay(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"   # the response ends when the connection closes, so streams need no chunk framing

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _body(self):
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            out = bytearray()
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    return bytes(out)
                out += self.rfile.read(size)
                self.rfile.readline()
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else None

    def _forward(self):
        try:
            body = self._body()
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "content-length"}
            headers["Host"] = UPSTREAM.netloc
            conn = upstream_conn()
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as exc:
            msg = ('{"error":"relay upstream %s: %s"}' % (UPSTREAM.netloc, str(exc).replace('"', "'")[:160])).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.end_headers()
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

    do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = do_PATCH = do_OPTIONS = _forward


if __name__ == "__main__":
    server = ThreadingHTTPServer((BIND_HOST, BIND_PORT), Relay)
    server.daemon_threads = True
    with open("/tmp/ollama_relay.pid", "w") as fh:
        fh.write(str(os.getpid()))
    print(f"Relay up: http://{BIND_HOST}:{BIND_PORT} -> {UPSTREAM.geturl()} (pid={os.getpid()})", flush=True)
    server.serve_forever()
