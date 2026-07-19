"""Background camera thread feeding frames to the UI.

Reads real webcam frames off the UI thread via the runtime's
``OpenCVCameraSource`` (the same capture boundary the pipeline uses) and emits
each frame as a Qt signal. Camera open failures are surfaced honestly rather than
silently showing a frozen image.
"""

from __future__ import annotations

from pathlib import Path
from queue import Empty, SimpleQueue
from typing import cast

import cv2
from PySide6.QtCore import QThread, Signal

from jarvis.calibration.profiles import load_profiles
from jarvis.gaze.engine import GazeTargetingEngine
from jarvis.gaze.classifier import DeviceGazeProfile
from jarvis.gaze.direction import CalibratedGaze, direction_to_yaw_pitch
from jarvis.gaze.landmarks import FaceLandmarkerAdapter, RgbFrame
from jarvis.monitoring.gaze_source import GazeSnapshot


class CameraWorker(QThread):
    frame_ready = Signal(object)  # numpy BGR frame
    gaze_ready = Signal(object)  # GazeSnapshot
    failed = Signal(str)
    gaze_failed = Signal(str)

    def __init__(
        self,
        device_index: int = 0,
        *,
        model_path: Path | None = None,
        profiles_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._device_index = device_index
        self._model_path = model_path
        self._profiles_path = profiles_path
        self._running = False
        self._profile_updates: SimpleQueue[tuple[str, DeviceGazeProfile | str]] = SimpleQueue()

    def register_profile(self, profile: DeviceGazeProfile) -> None:
        self._profile_updates.put(("upsert", profile))

    def unregister_profile(self, target_id: str) -> None:
        self._profile_updates.put(("remove", target_id))

    def _apply_profile_updates(self, engine: GazeTargetingEngine) -> None:
        while True:
            try:
                action, profile = self._profile_updates.get_nowait()
            except Empty:
                return
            if action == "upsert" and isinstance(profile, DeviceGazeProfile):
                engine.register_device(profile)
            elif isinstance(profile, str):
                engine.unregister_device(profile)

    def run(self) -> None:
        self._running = True
        source = cv2.VideoCapture(self._device_index)
        if not source.isOpened():
            source.release()
            self.failed.emit(f"카메라 {self._device_index}번을 열 수 없습니다")
            return
        gaze_adapter: FaceLandmarkerAdapter | None = None
        gaze_engine: GazeTargetingEngine | None = None
        if self._model_path is not None:
            try:
                gaze_adapter = FaceLandmarkerAdapter(self._model_path)
                gaze_engine = GazeTargetingEngine()
                if self._profiles_path is not None and self._profiles_path.is_file():
                    for profile in load_profiles(self._profiles_path):
                        gaze_engine.register_device(profile)
            except Exception as exc:  # noqa: BLE001 - keep camera alive, report gaze failure
                self.gaze_failed.emit(f"Gaze 초기화 실패: {exc}")
                gaze_adapter = None
                gaze_engine = None
        try:
            frame_id = 0
            while self._running:
                ok, image = source.read()
                if not ok:
                    self.msleep(5)
                    continue
                if gaze_adapter is not None and gaze_engine is not None:
                    self._apply_profile_updates(gaze_engine)
                    timestamp_ms = frame_id * 33
                    rgb_frame = cast(RgbFrame, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    observation = gaze_adapter.process(rgb_frame, timestamp_ms, frame_id)
                    estimate = gaze_engine.process(observation)
                    smoothed = gaze_engine.last_smoothed_gaze
                    calibrated = None
                    if smoothed is not None:
                        yaw, pitch = direction_to_yaw_pitch(smoothed.direction)
                        calibrated = CalibratedGaze(
                            yaw=yaw,
                            pitch=pitch,
                            confidence=min(
                                observation.eye_tracking_confidence,
                                observation.face_tracking_confidence,
                            ),
                            timestamp_ms=smoothed.timestamp_ms,
                        )
                    self.gaze_ready.emit(
                        GazeSnapshot(
                            observation=observation,
                            gaze_vector=smoothed,
                            estimate=estimate,
                            lock_state=str(gaze_engine.lock_state),
                            calibrated_gaze=calibrated,
                        )
                    )
                self.frame_ready.emit(image)
                frame_id += 1
        finally:
            if gaze_adapter is not None:
                gaze_adapter.close()
            source.release()

    def stop(self) -> None:
        self._running = False
        self.wait(2000)
