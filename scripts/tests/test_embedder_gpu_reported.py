"""ua-ingest.py and mnem_fix.py: no silent CPU / different-model embedder (EXT-6).

ua-ingest used to fall back to fastembed (a different build of the model, on CPU) whenever OLLAMA_URL was unset,
without a word; mnem_fix built a SentenceTransformer with no device, so it always ran on CPU. Now the fallback is an
explicit opt-in (UA_EMBED_FALLBACK=fastembed) that runs on CUDA and reports every miss, and mnem_fix asks for CUDA and
warns when torch cannot give it.

All collaborators (requests, fastembed, onnxruntime, sentence_transformers, torch) are recording fakes injected into
sys.modules; nothing here needs a GPU, a network or a model download.
"""
import importlib.util
import logging
import runpy
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
CUDA, CPU = "CUDAExecutionProvider", "CPUExecutionProvider"


# ── fakes ───────────────────────────────────────────────────────────────────

class FakeRequests(types.ModuleType):
    class HTTPError(Exception):
        pass

    def __init__(self, embeddings=None):
        super().__init__("requests")
        self.posts = []
        self._embeddings = embeddings or [[0.25, 0.5]]

    def post(self, url, json=None, timeout=None, **_kw):
        self.posts.append((url, json, timeout))
        outer = self

        class Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embeddings": outer._embeddings}

        return Resp()


class FakeVec:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)

    def __len__(self):
        return len(self._values)


def install_fastembed(monkeypatch, session_providers=(CUDA, CPU), expose_session=True):
    """Install fake `fastembed` + `onnxruntime`; return (constructions, preload_calls)."""
    constructions, preloads = [], []

    class TextEmbedding:
        def __init__(self, name, **kwargs):
            constructions.append((name, kwargs))
            if expose_session:
                self.model = types.SimpleNamespace(model=types.SimpleNamespace(get_providers=lambda: list(session_providers)))

        def embed(self, texts):
            return [FakeVec([float(len(t)), 1.0]) for t in texts]

    fe = types.ModuleType("fastembed")
    fe.TextEmbedding = TextEmbedding
    ort = types.ModuleType("onnxruntime")
    ort.preload_dlls = lambda: preloads.append("preload")
    ort.get_available_providers = lambda: [CUDA, CPU]
    monkeypatch.setitem(sys.modules, "fastembed", fe)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    return constructions, preloads


def load_ua(monkeypatch, requests_mod=None, **env):
    for key in ("OLLAMA_URL", "UA_EMBED_FALLBACK", "UA_CUDA_DEVICE", "UA_REQUIRE_CUDA", "UA_FASTEMBED_MODEL", "QDRANT_URL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    req = requests_mod if requests_mod is not None else FakeRequests()
    monkeypatch.setitem(sys.modules, "requests", req)
    spec = importlib.util.spec_from_file_location("ua_ingest_gpu_under_test", SCRIPTS / "ua-ingest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, req


# ── ua-ingest: backend selection ────────────────────────────────────────────

def test_ollama_url_selects_ollama_and_posts_the_texts_and_never_touches_fastembed(monkeypatch):
    constructions, preloads = install_fastembed(monkeypatch)
    ua, req = load_ua(monkeypatch, OLLAMA_URL="http://ollama.invalid:11434", EMBED_MODEL="nomic-embed-text")
    assert ua.resolve_backend() == "ollama"
    assert ua.embed_batch(["a", "bb"]) == [[0.25, 0.5]]
    assert req.posts == [("http://ollama.invalid:11434/api/embed", {"model": "nomic-embed-text", "input": ["a", "bb"]}, 120)]
    assert constructions == []
    assert preloads == []


def test_without_ollama_url_and_without_opt_in_there_is_no_embedder(monkeypatch):
    constructions, _ = install_fastembed(monkeypatch)
    ua, req = load_ua(monkeypatch)
    with pytest.raises(RuntimeError, match=r"OLLAMA_URL is not set, so there is no embedder\. Set OLLAMA_URL, or set UA_EMBED_FALLBACK=fastembed"):
        ua.embed_batch(["a"])
    assert constructions == []
    assert req.posts == []


def test_main_stops_before_touching_qdrant_when_there_is_no_embedder(monkeypatch, tmp_path):
    install_fastembed(monkeypatch)
    ua, req = load_ua(monkeypatch, QDRANT_URL="http://qdrant.invalid:6333")
    monkeypatch.setattr(sys, "argv", ["ua-ingest.py", str(tmp_path)])
    with pytest.raises(SystemExit, match="ERROR: OLLAMA_URL is not set, so there is no embedder"):
        ua.main()
    assert req.posts == []


# ── ua-ingest: the explicit fastembed fallback ──────────────────────────────

def test_opt_in_fallback_loads_fastembed_on_cuda_device_zero_after_preload_and_says_it_is_not_ollama(monkeypatch, caplog):
    constructions, preloads = install_fastembed(monkeypatch)
    ua, req = load_ua(monkeypatch, UA_EMBED_FALLBACK="fastembed")
    with caplog.at_level(logging.WARNING, logger="ua-ingest"):
        out = ua.embed_batch(["a", "bbb"])
    assert out == [[1.0, 1.0], [3.0, 1.0]]
    assert preloads == ["preload"]
    assert constructions == [("nomic-ai/nomic-embed-text-v1.5", {"providers": [(CUDA, {"device_id": 0}), CPU]})]
    assert [r.getMessage() for r in caplog.records] == [(
        "OLLAMA_URL is unset: embedding with fastembed 'nomic-ai/nomic-embed-text-v1.5' instead of Ollama 'nomic-embed-text' "
        "(UA_EMBED_FALLBACK=fastembed); vectors are not interchangeable with Ollama's")]
    assert dict(ua.DEGRADED) == {"fastembed-instead-of-ollama": 1}
    assert req.posts == []
    ua.embed_batch(["c"])
    assert len(constructions) == 1          # the model is loaded once per run


def test_fallback_device_comes_from_the_environment(monkeypatch):
    constructions, _ = install_fastembed(monkeypatch)
    ua, _req = load_ua(monkeypatch, UA_EMBED_FALLBACK="fastembed", UA_CUDA_DEVICE="1", UA_FASTEMBED_MODEL="some/other-model")
    ua.embed_batch(["a"])
    assert constructions == [("some/other-model", {"providers": [(CUDA, {"device_id": 1}), CPU]})]


def test_fallback_that_lands_on_cpu_is_reported_and_counted(monkeypatch, caplog):
    install_fastembed(monkeypatch, session_providers=(CPU,))
    ua, _req = load_ua(monkeypatch, UA_EMBED_FALLBACK="fastembed")
    with caplog.at_level(logging.WARNING, logger="ua-ingest"):
        out = ua.embed_batch(["a"])
    assert out == [[1.0, 1.0]]
    assert dict(ua.DEGRADED) == {"fastembed-instead-of-ollama": 1, "cuda-not-obtained": 1}
    assert caplog.records[-1].getMessage() == (
        "fastembed CUDA was requested (device_id=0) but the session runs on ['CPUExecutionProvider']; install fastembed-gpu / "
        "onnxruntime-gpu and the nvidia-cudnn wheels, or expect ~10x slower embedding")


def test_fallback_whose_session_cannot_be_inspected_is_reported_not_assumed_good(monkeypatch, caplog):
    install_fastembed(monkeypatch, expose_session=False)
    ua, _req = load_ua(monkeypatch, UA_EMBED_FALLBACK="fastembed")
    with caplog.at_level(logging.WARNING, logger="ua-ingest"):
        ua.embed_batch(["a"])
    assert dict(ua.DEGRADED) == {"fastembed-instead-of-ollama": 1, "cuda-not-obtained": 1}
    assert "runs on unknown providers (fastembed did not expose the session)" in caplog.records[-1].getMessage()


def test_require_cuda_refuses_a_cpu_fallback_session(monkeypatch):
    install_fastembed(monkeypatch, session_providers=(CPU,))
    ua, _req = load_ua(monkeypatch, UA_EMBED_FALLBACK="fastembed", UA_REQUIRE_CUDA="1")
    with pytest.raises(RuntimeError, match=r"fastembed CUDA was requested \(device_id=0\) but the session runs on \['CPUExecutionProvider'\]"):
        ua.embed_batch(["a"])


def test_require_cuda_accepts_a_cuda_session(monkeypatch):
    install_fastembed(monkeypatch)
    ua, _req = load_ua(monkeypatch, UA_EMBED_FALLBACK="fastembed", UA_REQUIRE_CUDA="1")
    assert ua.embed_batch(["a"]) == [[1.0, 1.0]]
    assert dict(ua.DEGRADED) == {"fastembed-instead-of-ollama": 1}


# ── mnem_fix ────────────────────────────────────────────────────────────────

def run_mnem_fix(monkeypatch, tmp_path, cuda_available=True, **env):
    """Run scripts/mnem_fix.py with fake torch + sentence_transformers and no databases; return the recorded constructions."""
    for key in ("MNEM_DEVICE", "MNEM_REQUIRE_CUDA"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["mnem_fix.py"])
    built = []

    class SentenceTransformer:
        def __init__(self, name, **kwargs):
            built.append((name, kwargs))

        def encode(self, text, **kwargs):
            return FakeVec([0.0] * 384)

    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = SentenceTransformer
    torch = types.ModuleType("torch")
    torch.__version__ = "2.14.1+cpu"
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.setitem(sys.modules, "torch", torch)
    runpy.run_path(str(SCRIPTS / "mnem_fix.py"), run_name="mnem_fix_under_test")
    return built


def test_mnem_fix_embeds_on_cuda_when_torch_has_it(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        built = run_mnem_fix(monkeypatch, tmp_path)
    assert built == [("BAAI/bge-small-en-v1.5", {"device": "cuda"})]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_mnem_fix_honours_an_explicit_cuda_index(monkeypatch, tmp_path):
    built = run_mnem_fix(monkeypatch, tmp_path, MNEM_DEVICE="cuda:1")
    assert built == [("BAAI/bge-small-en-v1.5", {"device": "cuda:1"})]


def test_mnem_fix_on_cpu_torch_falls_back_to_cpu_and_says_so(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        built = run_mnem_fix(monkeypatch, tmp_path, cuda_available=False)
    assert built == [("BAAI/bge-small-en-v1.5", {"device": "cpu"})]
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
        "MNEM_DEVICE='cuda' but torch 2.14.1+cpu has no usable CUDA; embedding on CPU (use an interpreter with CUDA torch)"]


def test_mnem_fix_require_cuda_refuses_cpu_torch(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match=r"MNEM_DEVICE='cuda' but torch 2\.14\.1\+cpu has no usable CUDA"):
        run_mnem_fix(monkeypatch, tmp_path, cuda_available=False, MNEM_REQUIRE_CUDA="1")


def test_mnem_fix_explicit_cpu_is_not_a_warning(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        built = run_mnem_fix(monkeypatch, tmp_path, cuda_available=False, MNEM_DEVICE="cpu")
    assert built == [("BAAI/bge-small-en-v1.5", {"device": "cpu"})]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
