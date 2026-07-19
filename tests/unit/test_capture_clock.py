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


def test_stamps_are_strictly_increasing_within_one_millisecond() -> None:
    """같은 ms 안에서 여러 프레임이 와도 timestamp는 엄격히 증가해야 한다.

    monotonic 시계의 ms 절삭으로 now_ms()가 반복돼도(FakeTime 고정), 다운스트림
    (MediaPipe detect_for_video·One-Euro)이 요구하는 strictly-increasing를 보장한다.
    """
    time_source = FakeTime(5_000_000)  # 5 ms에 고정 → now_ms()는 계속 5
    clock = RuntimeClock(time_source=time_source)
    stamps = [clock.stamp() for _ in range(4)]
    timestamps = [s.timestamp_ms for s in stamps]
    assert timestamps == [5, 6, 7, 8]  # 같은 ms지만 last+1로 강제
    assert all(b > a for a, b in zip(timestamps, timestamps[1:], strict=False))
    assert [s.frame_id for s in stamps] == [0, 1, 2, 3]


def test_stamp_follows_real_time_once_it_passes_the_bumped_value() -> None:
    """강제로 올린 값보다 실제 시간이 더 커지면 다시 실제 시간을 따른다(드리프트 없음)."""
    time_source = FakeTime(0)
    clock = RuntimeClock(time_source=time_source)
    first = clock.stamp()  # now=0 → 0
    second = clock.stamp()  # now=0 반복 → 1로 강제
    time_source.value_ns = 50_000_000  # 50 ms로 점프
    third = clock.stamp()  # 실제 50이 1보다 크므로 50
    assert [first.timestamp_ms, second.timestamp_ms, third.timestamp_ms] == [0, 1, 50]
