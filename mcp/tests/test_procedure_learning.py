"""Tests for procedure_learning — automatic promotion and execution feedback."""
import json
import sys
import tempfile
from pathlib import Path

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import procedure_learning as PL  # noqa: E402
import server  # noqa: E402
import verify as V  # noqa: E402


def _json(result: str) -> dict:
    return json.loads(result)


def _label_gen(label: str, ok: bool = True):
    def _stub(prompt, *, fmt=None, max_tokens=256):
        return {"text": label, "ok": ok}
    return _stub


def _verify_gen(verdict: str = "confirmed", confidence: float = 0.8):
    payload = json.dumps({
        "verdict": verdict,
        "reasoning": "tested",
        "refutation": "" if verdict == "confirmed" else "refuted",
        "confidence": confidence,
    })

    def _stub(prompt, *, fmt=None, max_tokens=256):
        return {"text": payload, "ok": True}
    return _stub


def _load_findings(inv_id: str) -> list[dict]:
    return server._read_jsonl(server._inv_dir(inv_id) / "findings.jsonl")


def _load_manifest(inv_id: str) -> dict:
    return json.loads((server._inv_dir(inv_id) / "manifest.json").read_text())


def _find_finding(inv_id: str, finding_id: str) -> dict:
    for finding in _load_findings(inv_id):
        if finding.get("id") == finding_id:
            return finding
    raise AssertionError(f"finding {finding_id} missing")


class TestProcedureLearning:
    def setup_method(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)
        server.inv_store.register(lambda: server.MEMORY_DIR)

    def teardown_method(self):
        server.MEMORY_DIR = self._orig
        server.inv_store.register(lambda: server.MEMORY_DIR)
        self._tmp.cleanup()

    def _start_and_store(self, *, finding_type: str = "observed", text: str) -> tuple[str, str]:
        inv_id = f"proc-learn-{finding_type}"
        server.investigation_start(investigation_id=inv_id, title="Procedure learning test")
        stored = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type=finding_type,
            text=text,
            source="unit-test",
            confidence="medium",
        ))
        return inv_id, stored["finding_id"]

    def test_action_shaped_claim_promotes(self):
        inv_id, finding_id = self._start_and_store(
            text="This incident is resolved by toggling the compatibility flag before retrying."
        )
        original = PL._heuristic_procedure_shape
        PL._heuristic_procedure_shape = lambda text: None

        try:
            result = PL.maybe_promote_to_procedure(
                investigation_id=inv_id,
                finding_id=finding_id,
                verify_verdict={"verdict": "confirmed", "degraded": False},
                gen_fn=_label_gen("procedure"),
            )
        finally:
            PL._heuristic_procedure_shape = original

        assert result["promoted"] is True
        finding = _find_finding(inv_id, finding_id)
        assert finding["record_type"] == "procedure"
        assert finding["type"] == "procedure"
        assert finding["procedure_meta"]["attempt_count"] == 0
        assert finding["procedure_meta"]["success_count"] == 0
        manifest = _load_manifest(inv_id)
        assert manifest["finding_counts"]["procedure"] == 1
        assert manifest["finding_counts"]["observed"] == 0

    def test_non_action_claim_does_not_promote(self):
        inv_id, finding_id = self._start_and_store(text="The logs show timeout spikes on replica 3.")

        result = PL.maybe_promote_to_procedure(
            investigation_id=inv_id,
            finding_id=finding_id,
            verify_verdict={"verdict": "confirmed", "degraded": False},
            gen_fn=_label_gen("non_procedure"),
        )

        assert result == {
            "promoted": False,
            "reason": "heuristic_non_action_shape",
            "degraded": False,
        }
        finding = _find_finding(inv_id, finding_id)
        assert finding["record_type"] == "observed"
        assert "procedure_meta" not in finding

    def test_already_procedure_is_not_double_promoted(self):
        inv_id, finding_id = self._start_and_store(
            finding_type="procedure",
            text="Drain the node before kernel upgrade.",
        )

        def _boom(*args, **kwargs):
            raise AssertionError("classifier must not be called for procedure findings")

        result = PL.maybe_promote_to_procedure(
            investigation_id=inv_id,
            finding_id=finding_id,
            verify_verdict="confirmed",
            gen_fn=_boom,
        )

        assert result == {"promoted": False, "reason": "already_procedure", "degraded": False}
        finding = _find_finding(inv_id, finding_id)
        assert finding["procedure_meta"]["attempt_count"] == 0
        assert finding["procedure_meta"]["success_count"] == 0

    def test_fail_open_on_model_error(self, monkeypatch):
        inv_id, finding_id = self._start_and_store(text="Applying the compatibility shim resolves the issue.")
        monkeypatch.setattr(PL, "_heuristic_procedure_shape", lambda text: None)

        def _raises(prompt, *, fmt=None, max_tokens=256):
            raise RuntimeError("ollama down")

        result = PL.maybe_promote_to_procedure(
            investigation_id=inv_id,
            finding_id=finding_id,
            verify_verdict={"verdict": "confirmed", "degraded": False},
            gen_fn=_raises,
        )

        assert result == {
            "promoted": False,
            "reason": "classifier_unavailable",
            "degraded": True,
        }
        finding = _find_finding(inv_id, finding_id)
        assert finding["record_type"] == "observed"

    def test_record_execution_outcome_delegates_to_procedure_attempt(self):
        inv_id, finding_id = self._start_and_store(text="Restart the worker to clear the stuck queue.")

        result = PL.record_execution_outcome(
            investigation_id=inv_id,
            finding_id=finding_id,
            success=True,
            source="auto-hook",
            gen_fn=_label_gen("procedure"),
        )

        assert result["recorded"] is True
        assert result["source"] == "auto-hook"
        assert result["attempt_count"] == 1
        assert result["success_count"] == 1
        finding = _find_finding(inv_id, finding_id)
        assert finding["record_type"] == "procedure"
        assert finding["procedure_meta"]["attempt_count"] == 1
        assert finding["procedure_meta"]["success_count"] == 1

    def test_verify_default_gate_leaves_behavior_unchanged(self, monkeypatch):
        inv_id, finding_id = self._start_and_store(text="Restart the API deployment to clear stale sockets.")
        called = {"value": False}

        def _spy(*args, **kwargs):
            called["value"] = True
            return {"promoted": True, "reason": "promoted", "degraded": False}

        monkeypatch.setattr(PL, "maybe_promote_to_procedure", _spy)

        result = V.verify_finding(
            "Restart the API deployment to clear stale sockets.",
            investigation_id=inv_id,
            finding_id=finding_id,
            gen_fn=_verify_gen(),
        )

        assert result["verdict"] == "confirmed"
        assert called["value"] is False
        finding = _find_finding(inv_id, finding_id)
        assert finding["record_type"] == "observed"

    def test_verify_opt_in_calls_auto_promotion(self, monkeypatch):
        inv_id, finding_id = self._start_and_store(text="Restart the API deployment to clear stale sockets.")
        calls = {}

        def _spy(*, investigation_id, finding_id, verify_verdict, gen_fn=None):
            calls["investigation_id"] = investigation_id
            calls["finding_id"] = finding_id
            calls["verdict"] = verify_verdict["verdict"]
            return {"promoted": True, "reason": "promoted", "degraded": False}

        monkeypatch.setattr(PL, "maybe_promote_to_procedure", _spy)

        result = V.verify_finding(
            "Restart the API deployment to clear stale sockets.",
            investigation_id=inv_id,
            finding_id=finding_id,
            gen_fn=_verify_gen(),
            auto_promote_procedures=True,
        )

        assert result["verdict"] == "confirmed"
        assert calls == {
            "investigation_id": inv_id,
            "finding_id": finding_id,
            "verdict": "confirmed",
        }
