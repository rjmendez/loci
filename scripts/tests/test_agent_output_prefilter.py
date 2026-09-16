import io
import json
import pathlib
import sys
import types
import builtins

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "mcp"))

import agent_output_prefilter as A  # noqa: E402
import embed_ops  # noqa: E402
import llm_local  # noqa: E402


class _FakeResp:
    def __init__(self, payload=None, raise_exc=None):
        self._payload = payload or {}
        self._raise = raise_exc

    def raise_for_status(self):
        if self._raise is not None:
            raise self._raise

    def json(self):
        return self._payload


def _install_fake_backends(monkeypatch):
    monkeypatch.setattr(embed_ops, "_resolve", lambda: ("http://fake-ollama:11434", "nomic-embed-text"))
    monkeypatch.setattr(llm_local, "_gen_env", lambda: "http://fake-ollama:11434")
    monkeypatch.setattr(llm_local, "_resolve_ollama", lambda: "http://fake-ollama:11434")
    monkeypatch.setattr(llm_local, "_try_vllm", lambda *a, **k: None)
    import backends

    monkeypatch.setattr(backends, "ollama_gen_model", lambda: "qwen2.5:3b")


def _install_requests(monkeypatch, *, embed=None, generate=None, capture=None):
    embed = list(embed or [])
    generate = list(generate or [])
    fake_exceptions = types.SimpleNamespace(
        Timeout=TimeoutError,
        ConnectionError=ConnectionError,
    )

    def fake_post(url, json=None, timeout=None):  # noqa: A002 - mirror requests kwarg
        if capture is not None:
            capture.append({"url": url, "json": json, "timeout": timeout})
        if url.endswith("/api/embed"):
            if not embed:
                raise AssertionError("unexpected embed call")
            item = embed.pop(0)
        elif url.endswith("/api/generate"):
            if not generate:
                raise AssertionError("unexpected generate call")
            item = generate.pop(0)
        else:
            raise AssertionError(f"unexpected URL: {url}")
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(post=fake_post, exceptions=fake_exceptions),
    )


def test_successful_prefilter_uses_relevance_then_compress(monkeypatch):
    _install_fake_backends(monkeypatch)
    calls = []
    _install_requests(
        monkeypatch,
        embed=[_FakeResp({"embeddings": [[1.0, 0.0], [0.9, 0.1]]})],
        generate=[_FakeResp({"response": "compressed summary with the build blocker"})],
        capture=calls,
    )
    text = "Build agent output. " * 120
    result = A.prefilter_agent_output(
        text, "summarize the build blocker", max_chars=160, chunk_chars=5000
    )

    assert result.fallback_used is False
    assert result.original_text == text
    assert result.compressed_text == "compressed summary with the build blocker"
    assert result.confidence >= 0.9
    assert 0.95 <= result.coverage <= 1.0
    assert [call["url"].rsplit("/", 1)[-1] for call in calls] == ["embed", "generate"]


def test_local_model_error_fails_open(monkeypatch, caplog):
    _install_fake_backends(monkeypatch)
    _install_requests(
        monkeypatch,
        embed=[_FakeResp({"embeddings": [[1.0, 0.0], [0.9, 0.1]]})],
        generate=[TimeoutError("timed out")],
    )
    text = "Research agent output. " * 120

    with caplog.at_level("WARNING", logger="loci-mcp.agent_output_prefilter"):
        result = A.prefilter_agent_output(
            text, "extract the key finding", max_chars=160, chunk_chars=5000
        )

    assert result.fallback_used is True
    assert result.compressed_text == text
    assert result.original_text == text
    assert "compress_text degraded" in result.fallback_reason
    assert "returning original text unchanged" in caplog.text


def test_low_confidence_fails_open(monkeypatch):
    _install_fake_backends(monkeypatch)
    calls = []
    _install_requests(
        monkeypatch,
        embed=[_FakeResp({"embeddings": [[1.0, 0.0], [0.5, 0.8660254]]})],
        generate=[],
        capture=calls,
    )
    text = "Explore agent output. " * 120
    result = A.prefilter_agent_output(
        text,
        "find the build failure",
        threshold=0.6,
        max_chars=160,
        chunk_chars=5000,
    )

    assert result.fallback_used is True
    assert result.compressed_text == text
    assert result.original_text == text
    assert "below threshold" in result.fallback_reason
    assert [call["url"].rsplit("/", 1)[-1] for call in calls] == ["embed"]


def test_nan_relevance_score_fails_open(monkeypatch):
    text = "Explore agent output. " * 120
    monkeypatch.setattr(
        A.llm_tools,
        "semantic_relevance",
        lambda chunks, context: '{"scores":[NaN],"degraded":false}',
    )

    def _unexpected_compress(*args, **kwargs):
        raise AssertionError("compress should not run for NaN relevance")

    monkeypatch.setattr(A._mcp_text_ops, "compress", _unexpected_compress)
    result = A.prefilter_agent_output(
        text,
        "find the build failure",
        threshold=0.6,
        max_chars=160,
        chunk_chars=5000,
    )

    assert result.fallback_used is True
    assert result.compressed_text == text
    assert result.original_text == text
    assert result.confidence == 0.0
    assert "no chunks cleared the semantic relevance cutoff" in result.fallback_reason


def test_cli_reads_stdin_and_emits_json(monkeypatch, capsys):
    _install_fake_backends(monkeypatch)
    _install_requests(
        monkeypatch,
        embed=[_FakeResp({"embeddings": [[1.0, 0.0], [0.9, 0.1]]})],
        generate=[_FakeResp({"response": "condensed answer"})],
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("Builder output. " * 120))

    rc = A.main(
        [
            "--context",
            "summarize the error",
            "--threshold",
            "0.6",
            "--max-chars",
            "160",
            "--chunk-chars",
            "5000",
        ]
    )

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["compressed_text"] == "condensed answer"
    assert out["fallback_used"] is False
    assert out["original_text"].startswith("Builder output.")
    assert set(out) >= {
        "compressed_text",
        "original_text",
        "confidence",
        "coverage",
        "fallback_used",
        "fallback_reason",
    }


def test_cli_invalid_utf8_file_input_does_not_crash(monkeypatch, capsys):
    class _BinaryFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"\xff\xfeBinary\x00text"

    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: _BinaryFile())

    rc = A.main(["--context", "summarize the bytes", "fake.bin"])

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["compressed_text"] == out["original_text"]
    assert out["original_text"].startswith("��Binary")
    assert out["fallback_used"] is False
