"""Frame sources: the boundary between hardware and the capture core.

A :class:`FrameSource` yields raw images. The core pipeline stamps and fans
them out; it does not care where they come from. Tests inject a fake source;
production uses :class:`OpenCVCameraSource`, the only part here that touches a
real device.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FrameSource(Protocol):
    """Yields raw images until exhausted or closed.

    ``read`` returns the next image, or ``None`` when no frame is available
    (end of stream or a transient camera miss). Implementations must be usable
    as a context manager so the pipeline can release the device deterministically.
    """

    def read(self) -> Any | None: ...

    def close(self) -> None: ...

    def __enter__(self) -> FrameSource: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


class OpenCVCameraSource:
    """Real webcam source backed by ``cv2.VideoCapture``.

    Hardware IO boundary (development-principles 1.1): it performs no stamping,
    fan-out, or success faking — it only reads real frames or reports failure.
    ``cv2`` is imported lazily so the capture core and its tests do not require
    the optional ``vision`` extra.
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._capture: Any | None = None

    def _ensure_open(self) -> Any:
        if self._capture is None:
            import cv2

            capture = cv2.VideoCapture(self._device_index)
            if not capture.isOpened():
                raise RuntimeError(
                    f"camera device {self._device_index} could not be opened"
                )
            self._capture = capture
        return self._capture

    def read(self) -> Any | None:
        ok, image = self._ensure_open().read()
        if not ok:
            return None
        return image

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> OpenCVCameraSource:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
