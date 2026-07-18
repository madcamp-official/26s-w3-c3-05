"""Unit tests for latency spans and aggregation."""

from __future__ import annotations

import pytest

from jarvis.runtime_protocol.telemetry.latency import (
    LatencyAggregator,
    LatencyStage,
    percentile,
    span_ms,
)


def test_span_is_end_minus_start() -> None:
    assert span_ms(100, 175) == 75


def test_span_rejects_negative() -> None:
    with pytest.raises(ValueError):
        span_ms(200, 100)


def test_percentile_nearest_rank() -> None:
    samples = [float(n) for n in range(1, 101)]  # 1..100
    assert percentile(samples, 50) == 50
    assert percentile(samples, 95) == 95
    assert percentile(samples, 99) == 99
    assert percentile(samples, 100) == 100
    assert percentile(samples, 0) == 1


def test_percentile_unsorted_input() -> None:
    assert percentile([30.0, 10.0, 20.0], 100) == 30.0


def test_percentile_of_empty_raises() -> None:
    with pytest.raises(ValueError):
        percentile([], 95)


def test_aggregator_rejects_negative_duration() -> None:
    agg = LatencyAggregator()
    with pytest.raises(ValueError):
        agg.record(LatencyStage.END_TO_END, -1)


def test_summary_none_before_any_sample() -> None:
    assert LatencyAggregator().summary(LatencyStage.END_TO_END) is None


def test_summary_aggregates_samples() -> None:
    agg = LatencyAggregator()
    for value in (100, 120, 140, 160, 500):
        agg.record(LatencyStage.END_TO_END, value)

    summary = agg.summary(LatencyStage.END_TO_END)
    assert summary is not None
    assert summary.count == 5
    assert summary.maximum == 500
    assert summary.mean == pytest.approx((100 + 120 + 140 + 160 + 500) / 5)
    assert summary.p95 == 500  # nearest-rank p95 of 5 samples → the top sample


def test_stages_are_independent() -> None:
    agg = LatencyAggregator()
    agg.record(LatencyStage.CAPTURE_TO_INFERENCE, 30)
    agg.record(LatencyStage.DISPATCH_TO_ACK, 800)

    summaries = agg.summaries()
    assert set(summaries) == {
        LatencyStage.CAPTURE_TO_INFERENCE,
        LatencyStage.DISPATCH_TO_ACK,
    }
    assert summaries[LatencyStage.CAPTURE_TO_INFERENCE].count == 1
    assert summaries[LatencyStage.DISPATCH_TO_ACK].maximum == 800


def test_count_tracks_recorded_samples() -> None:
    agg = LatencyAggregator()
    assert agg.count(LatencyStage.COMMIT_TO_DISPATCH) == 0
    agg.record(LatencyStage.COMMIT_TO_DISPATCH, 5)
    agg.record(LatencyStage.COMMIT_TO_DISPATCH, 7)
    assert agg.count(LatencyStage.COMMIT_TO_DISPATCH) == 2
