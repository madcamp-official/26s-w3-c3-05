"""The captured frame carried through the runtime.

``Frame`` is internal to ``runtime_protocol``; it is not part of the module
boundary contract. It pairs the shared :class:`FrameStamp` with an opaque image
payload so the core capture logic (clock, queue, fan-out) stays free of any
image library dependency and remains testable without a camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from jarvis.runtime_protocol.capture.clock import FrameStamp

ImageT = TypeVar("ImageT")


@dataclass(frozen=True, slots=True)
class Frame(Generic[ImageT]):
    """One captured image tagged with its shared-clock identity.

    ``image`` is whatever the frame source produces (e.g. a numpy ``ndarray``
    from OpenCV). The core keeps it opaque; only the camera boundary and the
    Gaze/Gesture consumers know its concrete type.
    """

    stamp: FrameStamp
    image: ImageT

    @property
    def timestamp_ms(self) -> int:
        return self.stamp.timestamp_ms

    @property
    def frame_id(self) -> int:
        return self.stamp.frame_id
