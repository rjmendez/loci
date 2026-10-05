"""Tests for advisory LLM wiring-obligation scanning."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server  # noqa: E402
import wiring_obligation_scan as W  # noqa: E402


def _json(result: str) -> dict:
    return json.loads(result)


def _gen_fn(text=None, *, ok=True, why=None):
    def _fn(prompt, model="", fmt=None, max_tokens=256, temperature=0.0, keep_alive="30m"):
        assert fmt == "json"
        assert "Code to inspect:" in prompt
        if not ok:
            return {"text": text or "", "ok": False, "model": model, "why": why or "not ok"}
        return {"text": text or "", "ok": True, "model": model}

    return _fn


def test_detects_clear_undeclared_obligation_pattern():
    result = W.scan(
        """
class LeaseManager:
    def acquire(self):
        \"\"\"Acquire the lease. Caller must call release() in a finally block.\"\"\"
        self._lock.acquire()
        self._leased = True
        return self
""",
        path="lease_manager.py",
        gen_fn=_gen_fn(
            """{
  "candidates": [
    {
      "description": "LeaseManager.acquire() acquires a lease that callers must later release().",
      "evidence_excerpt": "Caller must call release() in a finally block.\\nself._lock.acquire()",
      "confidence": "high",
      "suggested_declare_args": {
        "class_name": "LeaseManager",
        "method_name": "acquire",
        "expected_effect": "release the acquired lease by calling release()"
      }
    }
  ]
}"""
        ),
    )

    assert result["degraded"] is False
    assert result["error"] is None
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["confidence"] == "high"
    assert "release()" in candidate["description"]
    assert candidate["suggested_declare_args"] == {
        "class_name": "LeaseManager",
        "method_name": "acquire",
        "expected_effect": "release the acquired lease by calling release()",
    }


def test_returns_empty_for_clean_code_with_no_implicit_obligations():
    result = W.scan(
        """
def add(a, b):
    return a + b
""",
        path="math_utils.py",
        gen_fn=_gen_fn('{"candidates": []}'),
    )

    assert result == {"candidates": [], "degraded": False, "error": None}


def test_model_error_fails_open():
    def _boom(*args, **kwargs):
        raise RuntimeError("connection refused")

    result = W.scan("def noop():\n    pass\n", gen_fn=_boom)
    assert result["candidates"] == []
    assert result["degraded"] is True
    assert "generate() raised" in result["error"]


_TOGGLE_CANDIDATES = json.dumps({"candidates": [{
    "description": "Toggle.enable() sets a flag that callers must later disable().",
    "evidence_excerpt": "# caller must later disable() this flag",
    "confidence": "high",
    "suggested_declare_args": {
        "class_name": "Toggle",
        "method_name": "enable",
        "expected_effect": "call disable() to clear the enabled flag",
    },
}]})


def test_scan_has_no_wiring_store_side_effects(monkeypatch):
    """The scan returns a real candidate (scripted model) and still writes nothing.

    Without an injected model the hermetic env yields 0 candidates, so an
    auto-declare of every candidate could not be observed; here there is one.
    """
    from loci_fakes import fake_lazy_generate

    gen = fake_lazy_generate(lambda _prompt: {"text": _TOGGLE_CANDIDATES, "ok": True, "model": "fake"})
    monkeypatch.setattr(W, "_lazy_generate", gen)
    tmp = tempfile.TemporaryDirectory()
    orig = server.MEMORY_DIR
    try:
        server.MEMORY_DIR = Path(tmp.name)
        inv_id = "wiring-scan-no-side-effects"
        server.investigation_start(investigation_id=inv_id, title="scan side effects")
        inv_dir = server.MEMORY_DIR / inv_id
        before = {p.name: p.read_bytes() for p in inv_dir.iterdir() if p.is_file()}

        result = _json(server.wiring_obligation_scan(
            content="""
class Toggle:
    def enable(self):
        # caller must later disable() this flag
        self.enabled = True
""",
            path="toggle.py",
        ))

        after = {p.name: p.read_bytes() for p in inv_dir.iterdir() if p.is_file()}
        listed = _json(server.wiring_obligation_list(investigation_id=inv_id))

        assert len(gen.prompts) == 1
        assert "Code to inspect:" in gen.prompts[0]
        assert result["degraded"] is False
        assert [c["suggested_declare_args"]["class_name"] for c in result["candidates"]] == ["Toggle"]
        # Advisory only: no file in the investigation changed, nothing declared.
        assert before == after
        assert listed["unresolved_count"] == 0
        assert listed["obligations"] == []
    finally:
        server.MEMORY_DIR = orig
        tmp.cleanup()


def test_declare_is_visible_to_the_list_used_above(tmp_path, monkeypatch):
    """Positive twin: the same list call does see a declared obligation, so the
    empty list above means "not declared", not "the list is blind"."""
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "wiring-scan-twin"
    server.investigation_start(investigation_id=inv_id, title="declare twin")
    declared = _json(server.wiring_obligation_declare(
        investigation_id=inv_id, class_name="Toggle", method_name="enable",
        expected_effect="call disable() to clear the enabled flag",
    ))
    listed = _json(server.wiring_obligation_list(investigation_id=inv_id))
    assert listed["unresolved_count"] == 1
    assert [o["id"] for o in listed["obligations"]] == [declared["finding_id"]]
