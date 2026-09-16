import os
import sys
import types
from unittest import mock

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
        A.CaseScore(latency_ms=12.0, json_ok=False, schema_ok=False, correct=False, transport_ok=False),
        A.CaseScore(latency_ms=15.0, json_ok=False, schema_ok=False, correct=False, transport_ok=False),
    ]
    row = A.summarize_scores("classify", "candidate", "abliterated:latest", scores, note="connection refused")
    assert row.available is False
    assert row.note == "connection refused"
    assert row.json_rate() is None
    assert row.avg_latency_ms() is None


def test_aggregate_rows_combines_available_task_metrics():
    rows = [
        A.SummaryRow(task="classify", arm="baseline", model="qwen2.5:3b", total=2, json_ok=2, schema_ok=1, correct=1, latencies_ms=[10.0, 20.0], available=True),
        A.SummaryRow(task="compress", arm="baseline", model="qwen2.5:3b", total=1, json_ok=1, schema_ok=1, correct=1, latencies_ms=[30.0], available=True),
    ]
    agg = A.aggregate_rows(rows, arm="baseline", model="qwen2.5:3b")
    assert agg.task == "all"
    assert agg.available is True
    assert agg.total == 3
    assert agg.json_ok == 3
    assert agg.schema_ok == 2
    assert agg.correct == 2
    assert agg.avg_latency_ms() == 20.0


def test_format_table_renders_na_for_unavailable_rows():
    row = A.SummaryRow(task="verify", arm="candidate", model="abliterated:latest", total=3, json_ok=0, schema_ok=0, correct=0, latencies_ms=[], available=False, note="endpoint unavailable")
    table = A.format_table([row])
    assert "verify" in table
    assert "candidate" in table
    assert "N/A" in table
    assert "correct" in table
    assert "endpoint unavailable" in table


def test_evaluate_task_uses_stubbed_calls_without_live_ollama():
    def fake_call(base_url, model, prompt, max_tokens):
        del base_url, model, prompt, max_tokens
        return A.CallResult(text='{"label":"bug"}', latency_ms=9.0, transport_ok=True)

    row = A.evaluate_task("classify", "baseline", "qwen2.5:3b", base_url="http://fake", call_fn=fake_call)
    assert row.available is True
    assert row.total == len(A.CLASSIFY_CASES)
    assert row.json_ok == len(A.CLASSIFY_CASES)
    assert row.correct == 1


def test_score_case_marks_wrong_but_well_formed_classify_answer_incorrect():
    case = A.CLASSIFY_CASES[0]
    score = A.score_case("classify", case, A.CallResult(text='{"label":"feature"}', latency_ms=5.0, transport_ok=True))
    assert score.json_ok is True
    assert score.schema_ok is True
    assert score.correct is False


def test_score_case_marks_right_classify_answer_correct():
    case = A.CLASSIFY_CASES[0]
    score = A.score_case("classify", case, A.CallResult(text='{"label":"bug"}', latency_ms=5.0, transport_ok=True))
    assert score.json_ok is True
    assert score.schema_ok is True
    assert score.correct is True


def test_score_case_marks_wrong_but_well_formed_verify_answer_incorrect():
    case = A.VERIFY_CASES[1]
    score = A.score_case("verify", case, A.CallResult(text='{"verdict":"confirmed","reasoning":"wrong"}', latency_ms=5.0, transport_ok=True))
    assert score.json_ok is True
    assert score.schema_ok is True
    assert score.correct is False


def _fake_requests_module(json_payload):
    """Build a stand-in `requests` module so tests don't need the real dependency installed.

    call_ollama() lazily does `import requests` inside the function; injecting a fake module
    into sys.modules lets us exercise that code path without requiring the `requests` package
    to be present in every CI job that runs this test file.
    """
    fake_response = mock.Mock()
    fake_response.raise_for_status = mock.Mock()
    fake_response.json.return_value = json_payload
    fake_module = types.ModuleType("requests")
    fake_module.post = mock.Mock(return_value=fake_response)
    return fake_module


def test_call_ollama_disables_thinking_mode_in_request_body():
    fake_module = _fake_requests_module({"response": '{"label":"bug"}'})
    with mock.patch.dict(sys.modules, {"requests": fake_module}):
        result = A.call_ollama("http://fake", "qwen3-thinking:latest", "prompt", max_tokens=32)
    assert result.transport_ok is True
    assert result.text == '{"label":"bug"}'
    sent_body = fake_module.post.call_args.kwargs["json"]
    assert sent_body["think"] is False
    assert sent_body["format"] == "json"


def test_call_ollama_falls_back_to_thinking_field_when_response_is_empty():
    """Reasoning models sometimes still route JSON into `thinking` even with think=False."""
    fake_module = _fake_requests_module({"response": "", "thinking": '{"label":"bug"}'})
    with mock.patch.dict(sys.modules, {"requests": fake_module}):
        result = A.call_ollama("http://fake", "qwen3-thinking:latest", "prompt", max_tokens=32)
    assert result.transport_ok is True
    assert result.text == '{"label":"bug"}'


def test_call_ollama_prefers_response_over_thinking_when_both_present():
    fake_module = _fake_requests_module({"response": '{"label":"feature"}', "thinking": "some reasoning trace"})
    with mock.patch.dict(sys.modules, {"requests": fake_module}):
        result = A.call_ollama("http://fake", "qwen3-thinking:latest", "prompt", max_tokens=32)
    assert result.text == '{"label":"feature"}'


def test_score_case_marks_compress_keyword_retention_as_correctness_proxy():
    case = A.COMPRESS_CASES[0]
    wrong = A.score_case("compress", case, A.CallResult(text='{"text":"Short summary about budget only."}', latency_ms=5.0, transport_ok=True))
    right = A.score_case(
        "compress",
        case,
        A.CallResult(
            text='{"text":"Keep findings reloadable with claim, context, timestamps, and file:line references."}',
            latency_ms=5.0,
            transport_ok=True,
        ),
    )
    assert wrong.schema_ok is True
    assert wrong.correct is False
    assert right.schema_ok is True
    assert right.correct is True
