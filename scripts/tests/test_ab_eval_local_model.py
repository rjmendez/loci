import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mcp"))

import ab_eval_local_model as A  # noqa: E402


def test_validate_classify_schema_accepts_allowed_label():
    case = {"labels": ["bug", "feature", "question"]}
    assert A.validate_schema("classify", {"label": "bug"}, case) is True
    assert A.validate_schema("classify", {"label": "banana"}, case) is False


def test_validate_compress_schema_enforces_char_budget():
    case = {"max_chars": 10}
    assert A.validate_schema("compress", {"text": "short"}, case) is True
    assert A.validate_schema("compress", {"text": "this is too long"}, case) is False


def test_validate_verify_schema_limits_verdict_set():
    assert A.validate_schema("verify", {"verdict": "confirmed"}, {}) is True
    assert A.validate_schema("verify", {"verdict": "maybe"}, {}) is False


def test_summarize_scores_marks_transport_failures_unavailable():
    scores = [
        A.CaseScore(latency_ms=12.0, json_ok=False, schema_ok=False, transport_ok=False),
        A.CaseScore(latency_ms=15.0, json_ok=False, schema_ok=False, transport_ok=False),
    ]
    row = A.summarize_scores("classify", "candidate", "abliterated:latest", scores, note="connection refused")
    assert row.available is False
    assert row.note == "connection refused"
    assert row.json_rate() is None
    assert row.avg_latency_ms() is None


def test_aggregate_rows_combines_available_task_metrics():
    rows = [
        A.SummaryRow(task="classify", arm="baseline", model="qwen2.5:3b", total=2, json_ok=2, schema_ok=1, latencies_ms=[10.0, 20.0], available=True),
        A.SummaryRow(task="compress", arm="baseline", model="qwen2.5:3b", total=1, json_ok=1, schema_ok=1, latencies_ms=[30.0], available=True),
    ]
    agg = A.aggregate_rows(rows, arm="baseline", model="qwen2.5:3b")
    assert agg.task == "all"
    assert agg.available is True
    assert agg.total == 3
    assert agg.json_ok == 3
    assert agg.schema_ok == 2
    assert agg.avg_latency_ms() == 20.0


def test_format_table_renders_na_for_unavailable_rows():
    row = A.SummaryRow(task="verify", arm="candidate", model="abliterated:latest", total=3, json_ok=0, schema_ok=0, latencies_ms=[], available=False, note="endpoint unavailable")
    table = A.format_table([row])
    assert "verify" in table
    assert "candidate" in table
    assert "N/A" in table
    assert "endpoint unavailable" in table


def test_evaluate_task_uses_stubbed_calls_without_live_ollama():
    def fake_call(base_url, model, prompt, max_tokens):
        del base_url, model, prompt, max_tokens
        return A.CallResult(text='{"label":"bug"}', latency_ms=9.0, transport_ok=True)

    row = A.evaluate_task("classify", "baseline", "qwen2.5:3b", base_url="http://fake", call_fn=fake_call)
    assert row.available is True
    assert row.total == len(A.CLASSIFY_CASES)
    assert row.json_ok == len(A.CLASSIFY_CASES)
