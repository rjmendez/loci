import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import bench_local_models as B  # noqa: E402


def test_percentile_interpolates_between_points():
    values = [10.0, 20.0, 30.0, 40.0]
    assert B.percentile(values, 50) == 25.0
    assert B.percentile(values, 90) == 37.0
    assert B.percentile(values, 99) == pytest.approx(39.7)


def test_summarize_numeric_reports_expected_fields():
    summary = B.summarize_numeric([10.0, 20.0, 30.0])
    assert summary["count"] == 3
    assert summary["min"] == 10.0
    assert summary["max"] == 30.0
    assert summary["mean"] == 20.0
    assert round(summary["stddev"], 6) == 10.0
    assert summary["p50"] == 20.0
    assert summary["p90"] == 28.0
    assert summary["p99"] == 29.8


def test_tokens_per_second_refuses_missing_or_invalid_inputs():
    assert B.tokens_per_second(None, 1000.0) is None
    assert B.tokens_per_second(12, None) is None
    assert B.tokens_per_second(12, 0.0) is None
    assert B.tokens_per_second(12, 500.0) == 24.0


def test_summarize_suite_handles_streaming_and_speedups_without_network():
    req_a = B.RequestMetrics(
        total_ms=1000.0,
        ttft_ms=120.0,
        output_tokens=50,
        prompt_tokens=10,
        output_chars=100,
        wall_tokens_per_sec=50.0,
        server_eval_tokens_per_sec=80.0,
        transport_ok=True,
        stream_used=True,
    )
    req_b = B.RequestMetrics(
        total_ms=1100.0,
        ttft_ms=150.0,
        output_tokens=55,
        prompt_tokens=10,
        output_chars=120,
        wall_tokens_per_sec=50.0,
        server_eval_tokens_per_sec=82.0,
        transport_ok=True,
        stream_used=True,
    )
    one = B.summarize_suite(
        concurrency=1,
        warmup_iterations=2,
        trials=[B.TrialMetrics(trial_index=0, concurrency=1, wall_ms=1000.0, requests=[req_a])],
        resident_before=B.ResidencySnapshot(resident=True, loaded_model_names=["qwen2.5:3b"], raw=None),
        resident_after=B.ResidencySnapshot(resident=True, loaded_model_names=["qwen2.5:3b"], raw=None),
        stream_requested=True,
    )
    four = B.summarize_suite(
        concurrency=4,
        warmup_iterations=2,
        trials=[B.TrialMetrics(trial_index=0, concurrency=4, wall_ms=1200.0, requests=[req_a, req_a, req_b, req_b])],
        resident_before=B.ResidencySnapshot(resident=True, loaded_model_names=["qwen2.5:3b", "llama3.1-agent:latest"], raw=None),
        resident_after=B.ResidencySnapshot(resident=True, loaded_model_names=["qwen2.5:3b"], raw=None),
        stream_requested=True,
    )

    rows = [one, four]
    B.attach_speedups(rows)

    assert one["streaming"]["ttft_measured"] is True
    assert one["latency_ms"]["ttft"]["p50"] == 120.0
    assert one["tokens"]["aggregate_tokens_per_sec"]["mean"] == 50.0
    assert four["tokens"]["aggregate_tokens_per_sec"]["mean"] == 175.0
    assert four["tokens"]["aggregate_tokens_per_sec"]["speedup_vs_concurrency_1"] == 3.5
    assert four["aggregate_requests_per_sec"]["mean"] == (4 / 1.2)


def test_summarize_suite_calls_out_missing_ttft_when_streaming_disabled():
    req = B.RequestMetrics(
        total_ms=900.0,
        ttft_ms=None,
        output_tokens=30,
        prompt_tokens=8,
        output_chars=80,
        wall_tokens_per_sec=33.3333333333,
        server_eval_tokens_per_sec=60.0,
        transport_ok=True,
        stream_used=False,
    )
    row = B.summarize_suite(
        concurrency=1,
        warmup_iterations=0,
        trials=[B.TrialMetrics(trial_index=0, concurrency=1, wall_ms=900.0, requests=[req])],
        resident_before=B.ResidencySnapshot(resident=False, loaded_model_names=[], raw=None),
        resident_after=B.ResidencySnapshot(resident=True, loaded_model_names=["qwen2.5:3b"], raw=None),
        stream_requested=False,
    )
    assert row["streaming"]["ttft_measured"] is False
    assert row["streaming"]["ttft_note"] == "streaming disabled, so TTFT is unavailable by design"
    assert row["latency_ms"]["ttft"]["count"] == 0
