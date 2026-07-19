"""Unit tests for the single-capture fan-out pipeline."""

from __future__ import annotations

from types import TracebackType

import pytest

from jarvis.runtime_protocol.capture.clock import RuntimeClock
from jarvis.runtime_protocol.capture.pipeline import CapturePipeline
from jarvis.runtime_protocol.capture.source import EndOfStream


class FakeTime:
    def __init__(self) -> None:
        self.value_ns = 0

    def __call__(self) -> int:
        return self.value_ns


class ListFrameSource:
    """Yields a fixed list of images, then raises ``EndOfStream``.

    A finite source: exhaustion is a true end of stream, not a transient miss.
    """

    def __init__(self, images: list[str]) -> None:
        self._images = list(images)
        self.closed = False

    def read(self) -> str | None:
        if not self._images:
            raise EndOfStream
        return self._images.pop(0)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "ListFrameSource":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class FlakyFrameSource:
    """Replays a script of transient misses (``None``) and images, then ends."""

    def __init__(self, script: list[str | None]) -> None:
        self._script = list(script)
        self.closed = False

    def read(self) -> str | None:
        if not self._script:
            raise EndOfStream
        return self._script.pop(0)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "FlakyFrameSource":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _drain(pipeline: CapturePipeline) -> None:
    """Run captures until the finite source signals end of stream."""
    try:
        while True:
            pipeline.run_once()
    except EndOfStream:
        pass


def test_one_capture_delivers_identical_stamp_to_all_consumers() -> None:
    pipeline = CapturePipeline(ListFrameSource(["img-a"]), RuntimeClock(FakeTime()))
    gaze = pipeline.add_consumer("gaze", capacity=2)
    gesture = pipeline.add_consumer("gesture", capacity=2)

    frame = pipeline.run_once()
    assert frame is not None

    gaze_frame = gaze.get_nowait()
    gesture_frame = gesture.get_nowait()
    assert gaze_frame is not None and gesture_frame is not None
    # Same frame object → identical stamp and image for both consumers.
    assert gaze_frame is gesture_frame
    assert gaze_frame.stamp == frame.stamp
    assert gaze_frame.image == "img-a"


def test_frame_ids_advance_across_captures() -> None:
    pipeline = CapturePipeline(
        ListFrameSource(["a", "b", "c"]), RuntimeClock(FakeTime())
    )
    consumer = pipeline.add_consumer("gaze", capacity=8)

    _drain(pipeline)

    ids = []
    while (frame := consumer.get_nowait()) is not None:
        ids.append(frame.frame_id)
    assert ids == [0, 1, 2]


def test_run_once_raises_end_of_stream_when_source_exhausted() -> None:
    pipeline = CapturePipeline(ListFrameSource([]), RuntimeClock(FakeTime()))
    with pytest.raises(EndOfStream):
        pipeline.run_once()


def test_transient_miss_does_not_stop_pipeline() -> None:
    # None (a dropped webcam frame) must be retried, not treated as end of stream.
    source = FlakyFrameSource([None, "img", None])
    pipeline = CapturePipeline(source, RuntimeClock(FakeTime()))
    consumer = pipeline.add_consumer("gaze", capacity=4)

    assert pipeline.run_once() is None  # transient miss: nothing distributed
    frame = pipeline.run_once()  # real frame follows the miss
    assert frame is not None and frame.image == "img"
    assert pipeline.run_once() is None  # another transient miss
    with pytest.raises(EndOfStream):
        pipeline.run_once()  # now truly exhausted

    assert consumer.get_nowait() is frame
    assert consumer.get_nowait() is None


def test_slow_consumer_drops_oldest_without_affecting_others() -> None:
    pipeline = CapturePipeline(
        ListFrameSource(["a", "b", "c"]), RuntimeClock(FakeTime())
    )
    slow = pipeline.add_consumer("slow", capacity=1)
    fast = pipeline.add_consumer("fast", capacity=8)

    _drain(pipeline)

    # Slow consumer kept only the latest frame and counted the drops.
    latest = slow.get_nowait()
    assert latest is not None and latest.image == "c"
    assert slow.get_nowait() is None
    assert slow.dropped == 2

    # Fast consumer still has the full sequence.
    fast_images = []
    while (frame := fast.get_nowait()) is not None:
        fast_images.append(frame.image)
    assert fast_images == ["a", "b", "c"]


def test_cannot_add_consumer_after_start() -> None:
    pipeline = CapturePipeline(ListFrameSource([]), RuntimeClock(FakeTime()))
    pipeline.start()
    try:
        raised = False
        try:
            pipeline.add_consumer("late", capacity=1)
        except RuntimeError:
            raised = True
        assert raised
    finally:
        pipeline.stop(timeout=1.0)


def test_duplicate_consumer_name_rejected() -> None:
    pipeline = CapturePipeline(ListFrameSource([]), RuntimeClock(FakeTime()))
    pipeline.add_consumer("gaze", capacity=1)
    raised = False
    try:
        pipeline.add_consumer("gaze", capacity=1)
    except ValueError:
        raised = True
    assert raised


def test_start_then_stop_drains_stream() -> None:
    source = ListFrameSource(["a", "b"])
    pipeline = CapturePipeline(source, RuntimeClock(FakeTime()))
    consumer = pipeline.add_consumer("gaze", capacity=8)

    pipeline.start()
    pipeline.stop(timeout=1.0)

    images = []
    while (frame := consumer.get_nowait()) is not None:
        images.append(frame.image)
    assert images == ["a", "b"]


def test_close_releases_source() -> None:
    source = ListFrameSource([])
    pipeline = CapturePipeline(source, RuntimeClock(FakeTime()))
    pipeline.close()
    assert source.closed is True


def test_stop_alone_does_not_release_source() -> None:
    source = ListFrameSource(["a"])
    pipeline = CapturePipeline(source, RuntimeClock(FakeTime()))
    pipeline.start()
    pipeline.stop(timeout=1.0)
    assert source.closed is False  # stop() ends the loop but keeps the device


def test_context_manager_closes_source() -> None:
    source = ListFrameSource(["a"])
    with CapturePipeline(source, RuntimeClock(FakeTime())) as pipeline:
        pipeline.add_consumer("gaze", capacity=2)
    assert source.closed is True
