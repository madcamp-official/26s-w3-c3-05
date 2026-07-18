"""Background camera thread feeding frames to the UI.

Reads real webcam frames off the UI thread via the runtime's
``OpenCVCameraSource`` (the same capture boundary the pipeline uses) and emits
each frame as a Qt signal. Camera open failures are surfaced honestly rather than
silently showing a frozen image.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from jarvis.runtime_protocol.capture.source import OpenCVCameraSource


class CameraWorker(QThread):
    frame_ready = Signal(object)  # numpy BGR frame
    failed = Signal(str)

    def __init__(self, device_index: int = 0) -> None:
        super().__init__()
        self._device_index = device_index
        self._running = False

    def run(self) -> None:
        self._running = True
        source = OpenCVCameraSource(self._device_index)
        try:
            source.__enter__()
        except Exception as exc:  # noqa: BLE001 - report any open failure to the UI
            self.failed.emit(f"카메라 {self._device_index}번을 열 수 없습니다: {exc}")
            return
        try:
            while self._running:
                image = source.read()
                if image is None:
                    self.msleep(5)
                    continue
                self.frame_ready.emit(image)
        finally:
            source.close()

    def stop(self) -> None:
        self._running = False
        self.wait(2000)
