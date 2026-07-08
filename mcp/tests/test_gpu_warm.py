"""Tests for scripts/gpu_warm.py — the GPU warm-keeper. HTTP is stubbed; NO live Ollama.

We inject a stub post_fn (the [pattern:injectable] contract) and assert:
  - both hot models (qwen2.5:3b + nomic-embed-text) get pinned with keep_alive set,
    hitting the right endpoints (/api/generate + /api/embed);
  - the degraded path when Ollama is unreachable (poster raises) fails open, never raises.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# gpu_warm lives under scripts/, a sibling of mcp/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

# gpu_warm reads OLLAMA_BASE_URL at import time; set it before import so the module is
# "configured" and exercises the real request-building path against our stub.
os.environ.setdefault("OLLAMA_BASE_URL", "http://stub-ollama:11434")

import gpu_warm as G  # noqa: E402


class _Resp:
    """Minimal stand-in for a requests.Response."""
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _RecordingPoster:
    """Stub post_fn that records calls and returns canned bodies keyed by URL."""
    def __init__(self):
        self.calls = []

    def __call__(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if url.endswith("/api/ps"):
            return _Resp({"models": [
                {"name": "qwen2.5:3b", "size_vram": 3200000000},
                {"name": "nomic-embed-text", "size_vram": 280000000},
            ]})
        if url.endswith("/api/embed"):
            return _Resp({"embeddings": [[0.0, 0.1]]})
        # /api/generate
        return _Resp({"response": "", "done": True})


def _raising_poster(url, json=None, timeout=None):
    raise ConnectionError("stub: ollama unreachable")


def test_pin_gen_hits_generate_with_keep_alive():
    poster = _RecordingPoster()
    r = G.pin_model("qwen2.5:3b", "gen", keep_alive="-1", post_fn=poster)
    assert r["ok"] is True
    assert r["model"] == "qwen2.5:3b"
    assert r["keep_alive"] == -1  # "-1" coerced to int -1 (indefinite)
    call = poster.calls[-1]
    assert call["url"].endswith("/api/generate")
    assert call["json"]["keep_alive"] == -1
    assert call["json"]["model"] == "qwen2.5:3b"
    assert call["json"]["options"]["num_predict"] == 0  # load-only pin


def test_pin_embed_hits_embed_with_keep_alive():
    poster = _RecordingPoster()
    r = G.pin_model("nomic-embed-text", "embed", keep_alive="30m", post_fn=poster)
    assert r["ok"] is True
    call = poster.calls[-1]
    assert call["url"].endswith("/api/embed")
    assert call["json"]["keep_alive"] == "30m"  # duration string passes through
    assert call["json"]["model"] == "nomic-embed-text"


def test_warm_once_pins_both_models():
    poster = _RecordingPoster()
    report = G.warm_once(keep_alive="-1", post_fn=poster, include_gpu=False)
    assert report["degraded"] is False
    pinned = {(p["model"], p["kind"]): p for p in report["pins"]}
    # both hot models pinned, keep_alive set on each
    assert ("qwen2.5:3b", "gen") in pinned
    assert ("nomic-embed-text", "embed") in pinned
    assert all(p["ok"] and p["keep_alive"] == -1 for p in report["pins"])
    # /api/ps residency reported
    names = {m["name"] for m in report["ps"]["models"]}
    assert {"qwen2.5:3b", "nomic-embed-text"} <= names
    # endpoints actually exercised
    urls = [c["url"] for c in poster.calls]
    assert any(u.endswith("/api/generate") for u in urls)
    assert any(u.endswith("/api/embed") for u in urls)


def test_pin_degraded_on_unreachable_ollama():
    r = G.pin_model("qwen2.5:3b", "gen", post_fn=_raising_poster)
    assert r["ok"] is False
    assert "error" in r  # captured, not raised


def test_warm_once_degraded_path_fails_open():
    # Poster raises for every call -> pins fail, ps fails, but nothing propagates.
    report = G.warm_once(keep_alive="-1", post_fn=_raising_poster, include_gpu=False)
    assert report["degraded"] is True
    assert all(p["ok"] is False for p in report["pins"])
    assert report["ps"]["ok"] is False
    # report is still well-formed (fail-open contract)
    assert "pins" in report and "ps" in report and "ts" in report


def test_main_oneshot_exits_zero_even_when_degraded(monkeypatch, capsys):
    # Force the degraded path by making the lazy poster unresolvable.
    monkeypatch.setattr(G, "_resolve_post", lambda pf: None)
    rc = G.main(["--no-gpu"])  # one-shot
    assert rc == 0  # fail-open: exit 0
    out = capsys.readouterr().out
    assert "DEGRADED" in out
