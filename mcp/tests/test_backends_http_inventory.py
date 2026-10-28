"""The generation model is discovered over HTTP, so a remote Ollama (WSL talking to a Windows host, no `ollama`
binary) resolves to what it really has instead of a tag nobody installed."""
import http.server
import json
import os
import subprocess
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import backends as B  # noqa: E402
import pytest  # noqa: E402

GB = 10**9


class _FakeOllama:
    """A real HTTP server answering the two Ollama endpoints the resolvers read."""

    def __init__(self):
        self.tags: list[dict] = []
        self.ps: list[dict] = []
        self.paths: list[str] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.paths.append(self.path)
                payload = {"/api/tags": {"models": outer.tags}, "/api/ps": {"models": outer.ps}}.get(self.path)
                if payload is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def ollama(monkeypatch):
    server = _FakeOllama()
    cli_calls: list = []

    def cli(*args, **kwargs):
        cli_calls.append(args)
        raise FileNotFoundError("ollama")

    monkeypatch.setattr(subprocess, "run", cli)
    monkeypatch.setenv("LOCI_OLLAMA_GEN_URL", server.url)
    for var in ("LOCI_OLLAMA_GEN_MODEL", "LOCI_OLLAMA_VERIFY_MODEL", "LOCI_OLLAMA_AUTO_MAX_GB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(B, "_CONFIG_PATH", "/nonexistent")
    monkeypatch.setattr(B, "_pool_pick", lambda role: "")
    monkeypatch.setattr(B, "_warned_pinned", False)
    B._reset_cache()
    server.cli_calls = cli_calls
    yield server
    server.close()
    B._reset_cache()


def _tag(name, size):
    return {"name": name, "size": size}


def test_inventory_comes_from_api_tags_without_shelling_out(ollama):
    ollama.tags = [_tag("resident-chat:4b", 3_200_000_000), _tag("nomic-embed-text:latest", 274_000_000)]

    assert B._ollama_list() == {"resident-chat:4b": 3_200_000_000, "nomic-embed-text:latest": 274_000_000}
    assert B._ollama_local_tags() == {"resident-chat:4b", "nomic-embed-text:latest"}
    assert ollama.cli_calls == []


def test_the_cli_is_used_when_the_endpoint_does_not_answer(ollama, monkeypatch):
    out = "NAME ID SIZE MODIFIED\nqwen2.5:3b def456 1.9 GB 3 weeks ago\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, out, ""))
    monkeypatch.setenv("LOCI_OLLAMA_GEN_URL", "http://127.0.0.1:1")
    B._reset_cache()

    assert B._ollama_list() == {"qwen2.5:3b": 1_900_000_000}


def test_the_resident_model_beats_an_alphabetically_earlier_installed_one(ollama):
    ollama.tags = [_tag("aaa-small:1b", 1_300_000_000), _tag("resident-chat:4b", 3_200_000_000),
                   _tag("zzz-other:3b", 2_000_000_000)]
    ollama.ps = [{"name": "resident-chat:4b", "size_vram": 3_200_000_000}]

    assert B.ollama_gen_model() == "resident-chat:4b"
    assert B.ollama_verify_model() == "resident-chat:4b"


def test_with_nothing_resident_the_first_installed_generation_model_is_used(ollama):
    ollama.tags = [_tag("aaa-small:1b", 1_300_000_000), _tag("resident-chat:4b", 3_200_000_000),
                   _tag("nomic-embed-text:latest", 274_000_000)]
    ollama.ps = []

    assert B.ollama_gen_model() == "aaa-small:1b"


def test_a_resident_embedding_model_is_never_the_generation_model(ollama):
    ollama.tags = [_tag("small-chat:3b", 2_000_000_000), _tag("nomic-embed-text:latest", 274_000_000)]
    ollama.ps = [{"name": "nomic-embed-text:latest", "size_vram": 274_000_000}]

    assert B.ollama_gen_model() == "small-chat:3b"


def test_a_resident_model_over_the_one_gpu_cap_is_skipped(ollama, monkeypatch):
    ollama.tags = [_tag("gemma4:26b", 18_600_000_000), _tag("tiny:3b", 2_000_000_000)]
    ollama.ps = [{"name": "gemma4:26b", "size_vram": 18_600_000_000}]

    assert B.ollama_gen_model() == "tiny:3b"

    monkeypatch.setenv("LOCI_OLLAMA_AUTO_MAX_GB", "40")
    B._reset_cache()
    assert B.ollama_gen_model() == "gemma4:26b"


def test_the_pool_and_an_env_pin_both_outrank_the_resident_model(ollama, monkeypatch):
    ollama.tags = [_tag("resident-chat:4b", 3_200_000_000)]
    ollama.ps = [{"name": "resident-chat:4b", "size_vram": 3_200_000_000}]
    monkeypatch.setattr(B, "_pool_pick", lambda role: "pool-gen:4b")

    assert B.ollama_gen_model() == "pool-gen:4b"

    monkeypatch.setenv("LOCI_OLLAMA_GEN_MODEL", "env-gen:7b")
    assert B.ollama_gen_model() == "env-gen:7b"


def test_an_empty_first_lookup_backs_off_and_the_model_is_seen_after_the_retry_window(ollama, monkeypatch):
    """Main's RetryLatch is the cache: an empty inventory (HTTP empty, no CLI) is a failure, not an answer."""
    clock = [1000.0]
    monkeypatch.setattr(B, "_ollama_list_clock", lambda: clock[0])
    monkeypatch.setattr(B._ollama_list_latch, "_clock", lambda: clock[0])
    ollama.tags = []
    assert B._ollama_list() == {}

    ollama.tags = [_tag("late-arrival:3b", 2_000_000_000)]
    clock[0] += 1
    assert B._ollama_list() == {}

    clock[0] += 70
    assert B._ollama_list() == {"late-arrival:3b": 2_000_000_000}


def test_a_successful_http_inventory_is_served_from_the_cache_until_the_ttl_passes(ollama, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(B, "_ollama_list_clock", lambda: clock[0])
    monkeypatch.setattr(B._ollama_list_latch, "_clock", lambda: clock[0])
    ollama.tags = [_tag("first:3b", 2_000_000_000)]
    assert B._ollama_list() == {"first:3b": 2_000_000_000}

    ollama.tags = [_tag("second:3b", 3_000_000_000)]
    clock[0] += B._OLLAMA_LIST_TTL_S - 1
    assert B._ollama_list() == {"first:3b": 2_000_000_000}

    clock[0] += 2
    assert B._ollama_list() == {"second:3b": 3_000_000_000}
