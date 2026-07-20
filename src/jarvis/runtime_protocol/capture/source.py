"""Frame sources: the boundary between hardware and the capture core.

A :class:`FrameSource` yields raw images. The core pipeline stamps and fans
them out; it does not care where they come from. Tests inject a fake source;
production uses :class:`OpenCVCameraSource`, the only part here that touches a
real device.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Protocol, runtime_checkable


class EndOfStream(Exception):
    """Signals that a finite frame source will yield no further frames.

    This is distinct from a *transient* miss. :meth:`FrameSource.read` returns
    ``None`` when no frame is available **right now** (a live camera hiccup) and
    the pipeline should keep polling; it raises :class:`EndOfStream` only when no
    frame will **ever** arrive again (a finite replay/trace source), which stops
    the capture loop. Conflating the two would let a single dropped webcam frame
    silently kill the whole pipeline.
    """


@runtime_checkable
class FrameSource(Protocol):
    """Yields raw images until exhausted or closed.

    ``read`` returns the next image, or ``None`` on a **transient** miss (no
    frame available this instant — the caller should try again). A finite source
    raises :class:`EndOfStream` when it is exhausted. Implementations must be
    usable as a context manager so the pipeline can release the device
    deterministically.
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

    A live camera has no natural end of stream, so a failed read is reported as a
    transient miss (``None``); it never raises :class:`EndOfStream`. The pipeline
    stops the loop via :meth:`close`/``stop``, not via read failure.
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._capture: Any | None = None

    def _ensure_open(self) -> Any:
        if self._capture is None:
            import sys

            import cv2

            # Windows의 기본 백엔드(MSMF)는 내부적으로 프레임을 버퍼링해 지연이
            # 주기적으로 쌓였다 풀리는 끊김을 유발한다 — DSHOW로 이를 피한다.
            if sys.platform == "win32":
                capture = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)
            else:
                capture = cv2.VideoCapture(self._device_index)
            if not capture.isOpened():
                raise RuntimeError(
                    f"camera device {self._device_index} could not be opened"
                )
            # 버퍼를 1프레임으로 제한해, 소비자가 느려도 오래된 프레임이 쌓여
            # 나오지 않고(지연 누적 방지) 항상 최신 프레임을 읽게 한다.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
