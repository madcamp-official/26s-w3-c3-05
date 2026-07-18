"""Single monotonic clock and frame stamping for the whole runtime.

Every frame-based message on the module boundary inherits the ``timestamp_ms``
and ``frame_id`` issued here (interface-contract.md 공통 규칙, development-principles
4.3 / 5.1). Gaze and Gesture never mint their own timestamps; they carry the stamp
attached at capture time so that temporal alignment in Fusion is well defined.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Callable


@dataclass(frozen=True, slots=True)
class FrameStamp:
    """Identity of a single captured frame on the shared runtime clock."""

    timestamp_ms: int
    frame_id: int


class RuntimeClock:
    """Monotonic millisecond clock that also issues monotonic frame ids.

    A single instance is shared by the capture pipeline so that one captured
    frame yields exactly one ``FrameStamp`` propagated to all consumers.

    ``time_source`` returns nanoseconds and defaults to ``time.monotonic_ns``.
    It is injectable so tests can drive time deterministically without sleeping.
    """

    def __init__(self, time_source: Callable[[], int] = time.monotonic_ns) -> None:
        self._time_source = time_source
        self._lock = Lock()
        self._next_frame_id = 0

    def now_ms(self) -> int:
        """Current monotonic time in whole milliseconds."""
        return self._time_source() // 1_000_000

    def stamp(self) -> FrameStamp:
        """Issue the next frame stamp: current time plus a fresh frame id.

        Thread-safe: the capture thread calls this once per captured frame.
        Frame ids are strictly increasing and gapless within one process run.
        """
        with self._lock:
            frame_id = self._next_frame_id
            self._next_frame_id += 1
        return FrameStamp(timestamp_ms=self.now_ms(), frame_id=frame_id)
