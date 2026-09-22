"""Coverage for reflect aggregation preserving provenance and confidence metadata."""

from pathlib import Path
import sys

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import investigation_tools  # noqa: E402


def test_reflect_context_bullets_preserve_source_provenance_and_confidence():
    finding = {
        "id": "f-001",
        "type": "inferred",
        "text": "Replica drift was inferred from monotonic lag spikes.",
        "source": "unit:test",
        "confidence": "high",
        "derived_from": ["f-root"],
        "metadata": {"evidence_provenance_tier": "model_asserted"},
    }

    out = investigation_tools._reflect_context_bullets([finding], "inv-test")

    assert "id=f-001" in out
    assert "type=inferred" in out
    assert "source=unit:test" in out
    assert "confidence=high" in out
    assert "numeric_confidence=0.900" in out
    assert "provenance=model_asserted" in out
    assert "derived_from=f-root" in out


def test_reflect_context_bullets_default_legacy_provenance_when_missing():
    finding = {
        "id": "f-legacy",
        "type": "observed",
        "text": "Legacy finding with no explicit provenance tier.",
        "source": "unit:test",
        "confidence": "medium",
    }

    out = investigation_tools._reflect_context_bullets([finding], "inv-test")

    assert "provenance=tool_verified" in out
    assert "provenance_defaulted=true" in out
