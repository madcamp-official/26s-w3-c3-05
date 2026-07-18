"""Unit tests for the shared runtime clock and frame stamping."""

from __future__ import annotations

from jarvis.runtime_protocol.capture.clock import FrameStamp, RuntimeClock


class FakeTime:
    """Deterministic nanosecond time source driven by the test."""

    def __init__(self, start_ns: int = 0) -> None:
        self.value_ns = start_ns

    def __call__(self) -> int:
        return self.value_ns


def test_now_ms_converts_nanoseconds_to_whole_milliseconds() -> None:
    clock = RuntimeClock(time_source=FakeTime(1_500_000))  # 1.5 ms
    assert clock.now_ms() == 1


def test_frame_ids_are_gapless_and_increasing() -> None:
    clock = RuntimeClock(time_source=FakeTime())
    ids = [clock.stamp().frame_id for _ in range(4)]
    assert ids == [0, 1, 2, 3]


def test_stamp_uses_current_time() -> None:
    time_source = FakeTime()
    clock = RuntimeClock(time_source=time_source)

    time_source.value_ns = 10_000_000  # 10 ms
    first = clock.stamp()
    time_source.value_ns = 25_000_000  # 25 ms
    second = clock.stamp()

    assert first == FrameStamp(timestamp_ms=10, frame_id=0)
    assert second == FrameStamp(timestamp_ms=25, frame_id=1)


def test_default_clock_is_monotonic_non_decreasing() -> None:
    clock = RuntimeClock()
    readings = [clock.now_ms() for _ in range(5)]
    assert readings == sorted(readings)
