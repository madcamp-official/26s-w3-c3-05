"""End-to-end latency measurement and aggregation.

Latency is measured in the stages required by development-principles 5.4 and
README 13 (capture→inference, gesture-end→commit, commit→dispatch,
dispatch→ACK/verify), plus an overall end-to-end stage. Durations are computed
from two shared-clock timestamps and aggregated into percentiles so the p95
targets (노트북 ≤ 150ms, 전구 ≤ 1000ms) can be evaluated on real, reproducible
samples — never hand-made numbers (development-principles 1.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock


class LatencyStage(StrEnum):
    CAPTURE_TO_INFERENCE = "capture_to_inference"
    GESTURE_END_TO_COMMIT = "gesture_end_to_commit"
    COMMIT_TO_DISPATCH = "commit_to_dispatch"
    DISPATCH_TO_ACK = "dispatch_to_ack"
    END_TO_END = "end_to_end"


def span_ms(start_ms: int, end_ms: int) -> int:
    """Duration between two shared-clock timestamps.

    Raises ``ValueError`` on a negative span: end-before-start means an
    out-of-order or mis-clocked measurement, which must surface rather than
    silently corrupt the statistics.
    """
    duration = end_ms - start_ms
    if duration < 0:
        raise ValueError(f"negative span: start {start_ms} ms is after end {end_ms} ms")
    return duration


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Aggregate latency for one stage, in milliseconds."""

    count: int
    p50: float
    p95: float
    p99: float
    maximum: float
    mean: float


def percentile(samples: list[float], q: float) -> float:
    """Nearest-rank percentile of ``samples`` for ``q`` in [0, 100].

    Nearest-rank (rank = ceil(q/100 · n), 1-indexed) is unambiguous and matches
    how latency SLOs are usually read. ``samples`` need not be pre-sorted.
    """
    if not samples:
        raise ValueError("percentile of no samples")
    ordered = sorted(samples)
    if q <= 0:
        return ordered[0]
    if q >= 100:
        return ordered[-1]
    rank = math.ceil(q / 100 * len(ordered))
    return ordered[rank - 1]


class LatencyAggregator:
    """Collects per-stage duration samples and summarizes them."""

    def __init__(self) -> None:
        self._samples: dict[LatencyStage, list[float]] = {}
        self._lock = Lock()

    def record(self, stage: LatencyStage, duration_ms: float) -> None:
        """Record one duration sample for a stage. Rejects negative durations."""
        if duration_ms < 0:
            raise ValueError(f"duration must be >= 0, got {duration_ms}")
        with self._lock:
            self._samples.setdefault(stage, []).append(duration_ms)

    def count(self, stage: LatencyStage) -> int:
        with self._lock:
            return len(self._samples.get(stage, []))

    def summary(self, stage: LatencyStage) -> LatencySummary | None:
        """Aggregate a stage, or ``None`` if it has no samples yet."""
        with self._lock:
            samples = list(self._samples.get(stage, []))
        if not samples:
            return None
        return LatencySummary(
            count=len(samples),
            p50=percentile(samples, 50),
            p95=percentile(samples, 95),
            p99=percentile(samples, 99),
            maximum=max(samples),
            mean=sum(samples) / len(samples),
        )

    def summaries(self) -> dict[LatencyStage, LatencySummary]:
        """Summaries for every stage that has samples."""
        with self._lock:
            stages = list(self._samples.keys())
        result: dict[LatencyStage, LatencySummary] = {}
        for stage in stages:
            summary = self.summary(stage)
            if summary is not None:
                result[stage] = summary
        return result
