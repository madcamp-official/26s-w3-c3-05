"""Unit tests for the single-capture fan-out pipeline."""

from __future__ import annotations

from types import TracebackType

from jarvis.runtime_protocol.capture.clock import RuntimeClock
from jarvis.runtime_protocol.capture.pipeline import CapturePipeline


class FakeTime:
    def __init__(self) -> None:
        self.value_ns = 0

    def __call__(self) -> int:
        return self.value_ns


class ListFrameSource:
    """Yields a fixed list of images, then ``None`` (end of stream)."""

    def __init__(self, images: list[str]) -> None:
        self._images = list(images)
        self.closed = False

    def read(self) -> str | None:
        if not self._images:
            return None
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

    while pipeline.run_once() is not None:
        pass

    ids = []
    while (frame := consumer.get_nowait()) is not None:
        ids.append(frame.frame_id)
    assert ids == [0, 1, 2]


def test_run_once_returns_none_at_end_of_stream() -> None:
    pipeline = CapturePipeline(ListFrameSource([]), RuntimeClock(FakeTime()))
    assert pipeline.run_once() is None


def test_slow_consumer_drops_oldest_without_affecting_others() -> None:
    pipeline = CapturePipeline(
        ListFrameSource(["a", "b", "c"]), RuntimeClock(FakeTime())
    )
    slow = pipeline.add_consumer("slow", capacity=1)
    fast = pipeline.add_consumer("fast", capacity=8)

    while pipeline.run_once() is not None:
        pass

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
