"""Background camera thread feeding frames (and gaze snapshots) to the UI.

Reads real webcam frames off the UI thread via the runtime's
``OpenCVCameraSource`` (the same capture boundary the pipeline uses) and emits
each frame as a Qt signal. When a :class:`GazeProbe` is attached, each frame is
also run through the real gaze pipeline here (off the UI thread, since MediaPipe
inference is heavy) and the resulting snapshot is emitted. Camera open failures
are surfaced honestly rather than silently showing a frozen image.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QThread, Signal

from jarvis.monitoring.gaze_probe import GazeProbe
from jarvis.runtime_protocol.capture.source import OpenCVCameraSource


class CameraWorker(QThread):
    frame_ready = Signal(object)  # numpy BGR frame
    gaze_ready = Signal(object)  # GazeSnapshot (only when a probe is attached)
    failed = Signal(str)

    def __init__(self, device_index: int = 0, probe: GazeProbe | None = None) -> None:
        super().__init__()
        self._device_index = device_index
        self._probe = probe
        self._running = False

    def run(self) -> None:
        self._running = True
        source = OpenCVCameraSource(self._device_index)
        try:
            source.__enter__()
        except Exception as exc:  # noqa: BLE001 - report any open failure to the UI
            self.failed.emit(f"카메라 {self._device_index}번을 열 수 없습니다: {exc}")
            return
        start = time.monotonic()
        frame_id = 0
        try:
            while self._running:
                image = source.read()
                if image is None:
                    self.msleep(5)
                    continue
                self.frame_ready.emit(image)
                if self._probe is not None and self._probe.available:
                    timestamp_ms = int((time.monotonic() - start) * 1000)
                    try:
                        snapshot = self._probe.process_bgr(image, timestamp_ms, frame_id)
                    except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the thread
                        self.failed.emit(f"gaze 처리 오류: {exc}")
                        snapshot = None
                    if snapshot is not None:
                        self.gaze_ready.emit(snapshot)
                    frame_id += 1
        finally:
            if self._probe is not None:
                self._probe.close()
            source.close()

    def stop(self) -> None:
        self._running = False
        self.wait(2000)
