"""Single-capture fan-out pipeline.

One frame is captured once, stamped once on the shared clock, and delivered to
every consumer (Gaze, Gesture, ...) so they align on identical
``timestamp_ms``/``frame_id`` (development-principles 5.1). Each consumer has its
own bounded latest-frame queue, so a slow consumer drops its own stale frames
without affecting the others.
"""

from __future__ import annotations

from threading import Event, Thread
from types import TracebackType
from typing import Any

from jarvis.runtime_protocol.capture.clock import RuntimeClock
from jarvis.runtime_protocol.capture.frame import Frame
from jarvis.runtime_protocol.capture.queue import BoundedLatestQueue
from jarvis.runtime_protocol.capture.source import EndOfStream, FrameSource


class CapturePipeline:
    """Pulls frames from a source, stamps them, and fans out to consumers.

    Consumers are registered up front with :meth:`add_consumer`, which returns
    the queue that consumer reads from. :meth:`run_once` performs one full
    capture→stamp→distribute step and is the deterministic unit exercised by
    tests. :meth:`start`/:meth:`stop` run that step on a background thread.
    """

    def __init__(self, source: FrameSource, clock: RuntimeClock) -> None:
        self._source = source
        self._clock = clock
        self._consumers: dict[str, BoundedLatestQueue[Frame[Any]]] = {}
        self._stop = Event()
        self._thread: Thread | None = None

    def add_consumer(self, name: str, capacity: int) -> BoundedLatestQueue[Frame[Any]]:
        """Register a consumer and return its bounded queue.

        Raises if two consumers share a name or if called after :meth:`start`.
        """
        if self._thread is not None:
            raise RuntimeError("cannot add a consumer after the pipeline has started")
        if name in self._consumers:
            raise ValueError(f"consumer {name!r} is already registered")
        queue: BoundedLatestQueue[Frame[Any]] = BoundedLatestQueue(capacity)
        self._consumers[name] = queue
        return queue

    def run_once(self) -> Frame[Any] | None:
        """Capture, stamp, and distribute one frame.

        Returns the distributed :class:`Frame`, or ``None`` on a **transient**
        miss (no frame available this tick — the caller should try again). Raises
        :class:`EndOfStream` when a finite source is exhausted. The same
        ``Frame`` object is delivered to every consumer so their stamps are
        identical by construction.
        """
        image = self._source.read()
        if image is None:
            return None
        frame: Frame[Any] = Frame(stamp=self._clock.stamp(), image=image)
        for queue in self._consumers.values():
            queue.put(frame)
        return frame

    def start(self) -> None:
        """Begin capturing on a background thread until stopped or exhausted."""
        if self._thread is not None:
            raise RuntimeError("pipeline is already running")
        self._stop.clear()
        thread = Thread(target=self._loop, name="capture-pipeline", daemon=True)
        self._thread = thread
        thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Signal the capture loop to stop and join the thread.

        Does not release the frame source; call :meth:`close` (or use the
        pipeline as a context manager) to release the device.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def close(self) -> None:
        """Stop capturing and release the frame source's device.

        Safe to call more than once. This is what actually frees a real webcam;
        without it the underlying ``cv2.VideoCapture`` stays open for the life of
        the process.
        """
        self.stop()
        self._source.close()

    def __enter__(self) -> CapturePipeline:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _loop(self) -> None:
        # A transient miss (``run_once`` returns None) is retried; only a finite
        # source raising EndOfStream — or an explicit stop — ends the loop.
        while not self._stop.is_set():
            try:
                self.run_once()
            except EndOfStream:
                break
