"""Run MediaPipe hand landmark detection on frames for the monitor.

This is the wiring layer for the Gesture module's *vision half*. It runs the real
MediaPipe Hand Landmarker per frame and reuses gesture_fusion's public
normalization (``RawHandLandmarks`` → ``normalize_hand`` → ``HandObservation``),
so the values shown are the exact ones downstream would consume.

Honesty scope (development-principles 1.1, and gesture-fusion.md Task 3 note):
the gesture *recognition* model (Causal TCN) is **untrained** (random weights,
``ModelMetadata.trained=False``) and needs the ``ml`` extra (torch), which is not
required here. So this probe deliberately does **hand tracking only** — hand
presence, handedness, detection confidence, landmark geometry — and never emits a
recognized gesture. Feeding an untrained model's output into the UI as a
"recognized gesture" would fabricate a result, which this project forbids.

The MediaPipe Hand Landmarker returns image-space landmark coordinates; this probe
keeps them (for drawing the skeleton on the webcam) in addition to producing the
normalized ``HandObservation``. Owning the landmarker here (rather than reusing
``MediaPipeHandLandmarker``, which discards image coordinates) is exactly the
capture↔vision wiring responsibility gesture-fusion.md assigns to this layer.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from jarvis.gesture_fusion.config import DEFAULT_GESTURE_CONFIG, GestureConfig
from jarvis.gesture_fusion.landmarks import RawHandLandmarks, normalize_hand


@dataclass(frozen=True, slots=True)
class HandSnapshot:
    """Real hand-tracking result for one frame (no gesture recognition)."""

    timestamp_ms: int
    frame_id: int
    hand_detected: bool
    handedness: str
    handedness_score: float
    detection_confidence: float
    palm_scale: float
    # 21 (x, y) image-normalized coords in [0, 1] for the webcam overlay; None when
    # no hand is tracked (never a stale/faked skeleton).
    image_points: tuple[tuple[float, float], ...] | None
    landmark_count: int
    inference_ms: float


def _gesture_recognition_status() -> str:
    """Honest one-liner: why gesture *recognition* is off even though hands track.

    Names the two real reasons — the classifier model is untrained, and the
    trained-weights file / torch are absent — so the UI never implies gestures
    are being recognized.
    """
    torch_present = importlib.util.find_spec("torch") is not None
    weights = Path("models/gesture_tcn.pt")
    parts = ["제스처 인식 모델 미학습(무작위 가중치, trained=False)"]
    if not torch_present:
        parts.append("torch(ml extra) 미설치")
    if not weights.is_file():
        parts.append("학습 가중치(models/gesture_tcn.pt) 없음")
    return " · ".join(parts) + " — 인식 비활성 (손 추적만 라이브)"


class HandProbe:
    """Owns the live MediaPipe Hand Landmarker and turns BGR frames into snapshots.

    The landmarker is created lazily, so this class can be constructed and its
    liveness checked without the ``vision`` extra or a model file present.
    """

    def __init__(
        self,
        *,
        model_path: Path | None,
        config: GestureConfig = DEFAULT_GESTURE_CONFIG,
    ) -> None:
        self._model_path = model_path
        self._config = config
        self._landmarker: object | None = None
        self._available = False
        self._status_text = "hand 프로브 미시작"
        self._gesture_status = _gesture_recognition_status()

    @property
    def available(self) -> bool:
        return self._available

    @property
    def status_text(self) -> str:
        return self._status_text

    @property
    def gesture_recognition_status(self) -> str:
        return self._gesture_status

    def start(self) -> bool:
        """Create the MediaPipe Hand Landmarker. Returns True on success.

        Missing mediapipe or model file sets an honest ``status_text`` and leaves
        the probe unavailable — it never pretends to track.
        """
        if self._model_path is None or not self._model_path.is_file():
            self._status_text = "hand_landmarker.task 모델 없음 (models/README.md 참고)"
            return False
        try:
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )
        except ImportError:
            self._status_text = "mediapipe 미설치 — pip install -e \".[vision]\""
            return False
        try:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self._model_path)),
                running_mode=RunningMode.VIDEO,
                num_hands=self._config.num_hands,
                min_hand_detection_confidence=self._config.min_hand_detection_confidence,
                min_hand_presence_confidence=self._config.min_hand_presence_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
        except Exception as exc:  # noqa: BLE001 - surface any init failure honestly
            self._status_text = f"hand 랜드마커 초기화 실패: {exc}"
            return False
        self._available = True
        self._status_text = f"LIVE · {self._model_path.name} (손 추적)"
        return True

    def process_bgr(
        self, bgr_frame: npt.NDArray[np.uint8], timestamp_ms: int, frame_id: int
    ) -> HandSnapshot | None:
        """Convert a BGR frame, run hand detection, return a snapshot (or None).

        Returns ``None`` when the probe is not available. Tracking loss / low
        confidence yields ``hand_detected=False`` rather than an invented pose.
        """
        if self._landmarker is None:
            return None
        import time

        import cv2
        from mediapipe import Image as MpImage
        from mediapipe import ImageFormat as MpImageFormat
        from mediapipe.tasks.python.vision import HandLandmarker

        assert isinstance(self._landmarker, HandLandmarker)
        started = time.monotonic()
        rgb = cast("npt.NDArray[np.uint8]", cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        mp_image = MpImage(image_format=MpImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        inference_ms = (time.monotonic() - started) * 1000.0

        if not result.hand_landmarks:
            return self._lost(timestamp_ms, frame_id, inference_ms)

        landmarks = result.hand_landmarks[0]
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
        if points.shape != (21, 3):
            return self._lost(timestamp_ms, frame_id, inference_ms)

        handedness, score = self._primary_handedness(result)
        raw = RawHandLandmarks(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            points=points,
            handedness=handedness,
            detection_confidence=score,
            handedness_score=score,
        )
        observation = normalize_hand(raw, self._config)
        if not observation.hand_detected:
            return self._lost(timestamp_ms, frame_id, inference_ms)

        image_points = tuple((float(p[0]), float(p[1])) for p in points)
        return HandSnapshot(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            hand_detected=True,
            handedness=observation.handedness,
            handedness_score=observation.handedness_score,
            detection_confidence=observation.detection_confidence,
            palm_scale=observation.palm_scale,
            image_points=image_points,
            landmark_count=len(image_points),
            inference_ms=inference_ms,
        )

    @staticmethod
    def _primary_handedness(result: object) -> tuple[str, float]:
        handedness_list = getattr(result, "handedness", None)
        if not handedness_list or not handedness_list[0]:
            return "", 0.0
        top = handedness_list[0][0]
        return str(top.category_name), float(top.score)

    @staticmethod
    def _lost(timestamp_ms: int, frame_id: int, inference_ms: float) -> HandSnapshot:
        return HandSnapshot(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            hand_detected=False,
            handedness="",
            handedness_score=0.0,
            detection_confidence=0.0,
            palm_scale=0.0,
            image_points=None,
            landmark_count=0,
            inference_ms=inference_ms,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            from mediapipe.tasks.python.vision import HandLandmarker

            assert isinstance(self._landmarker, HandLandmarker)
            self._landmarker.close()
            self._landmarker = None
            self._available = False
