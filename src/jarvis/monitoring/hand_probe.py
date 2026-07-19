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

from jarvis.gesture_fusion.config import DEFAULT_GESTURE_CONFIG, LANDMARK_DIMS, GestureConfig
from jarvis.gesture_fusion.features import HandFeatureExtractor
from jarvis.gesture_fusion.landmarks import RawHandLandmarks, normalize_hand
from jarvis.gesture_fusion.smoothing import OneEuroFilter

Point2D = tuple[float, float]


def _as_vec2(vec: npt.NDArray[np.float64] | None) -> tuple[float, float] | None:
    """Convert a length-2 array to a plain float tuple for the UI (None passes through)."""
    if vec is None:
        return None
    return (float(vec[0]), float(vec[1]))


@dataclass(frozen=True, slots=True)
class HandSnapshot:
    """Real hand-tracking result for one frame (no gesture recognition).

    The debugging view distinguishes two coordinate spaces:
    - ``image_points``: raw detection in image space [0, 1], for locating the hand
      on the webcam. This is *not* what the model sees (position/scale intact).
    - ``model_points``: the exact normalized landmarks the model consumes this
      frame — smoothed when smoothing is on — so the display equals the model
      input rather than a separate approximation. Wrist-origin, palm-scaled.
    """

    timestamp_ms: int
    frame_id: int
    hand_detected: bool
    handedness: str
    handedness_score: float
    detection_confidence: float
    palm_scale: float
    # raw image-space detection (x, y) in [0, 1] for the webcam overlay; None when lost.
    image_points: tuple[Point2D, ...] | None
    # the actual model input this frame: (x, y) of the smoothed normalized landmarks.
    model_points: tuple[Point2D, ...] | None
    # the same normalized landmarks BEFORE smoothing (for the raw/smoothed compare).
    model_points_raw: tuple[Point2D, ...] | None
    landmark_count: int
    inference_ms: float
    # whether ``model_points`` reflects the smoothing the model actually applies.
    smoothed: bool
    # The wrist-translation feature the model consumes this frame: normalized
    # (palm-scaled, camera-distance independent) wrist velocity and acceleration,
    # each (x, y) in palm-widths/second (z dropped — config.LANDMARK_DIMS). None when
    # lost or on the first frame after a reset (no causal history yet). This is the
    # signal that makes a pure translation (swipe) visible even though wrist-origin
    # normalization zeroes it out of the landmark block.
    wrist_velocity: tuple[float, float] | None
    wrist_acceleration: tuple[float, float] | None
    # One-Euro-smoothed image-space (x, y) in [0, 1] for the live webcam overlay.
    # None when smoothing is off or the hand is lost. Display-only — this is *never*
    # fed to the model or logged for training (that path uses the raw ``points``).
    image_points_smoothed: tuple[Point2D, ...] | None = None


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
        smoothing: bool = True,
    ) -> None:
        self._model_path = model_path
        self._config = config
        self._landmarker: object | None = None
        self._available = False
        self._status_text = "hand 프로브 미시작"
        self._gesture_status = _gesture_recognition_status()
        self._smoothing = smoothing
        # Run the model's real feature extractor so the displayed model_points are
        # the exact normalized landmarks the model consumes (not a parallel filter).
        self._extractor = HandFeatureExtractor(config)
        # Display-only One-Euro filter for the live webcam overlay's image-space
        # points. Kept separate from the model's normalized-space smoothing
        # (``self._extractor``): it only de-jitters the skeleton drawn on the webcam
        # and never touches the landmarks fed to the model or logged for training.
        self._image_smoother: OneEuroFilter | None = (
            OneEuroFilter(
                min_cutoff=config.smoothing_min_cutoff,
                beta=config.smoothing_beta,
                d_cutoff=config.smoothing_d_cutoff,
            )
            if config.smooth_landmarks
            else None
        )

    @property
    def available(self) -> bool:
        return self._available

    @property
    def smoothing(self) -> bool:
        return self._smoothing

    def set_smoothing(self, enabled: bool) -> None:
        """Toggle which model input the display shows: smoothed (real) or raw."""
        self._smoothing = enabled

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
        # z(깊이)는 단안 웹캠 추정값이라 노이즈가 커 버리고 x·y만 쓴다(config.LANDMARK_DIMS).
        points = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float64)
        if points.shape != (21, LANDMARK_DIMS):
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

        # Feed the model's real feature extractor so ``last_landmarks`` is the exact
        # normalized landmark set the model consumes this frame (smoothed if enabled).
        self._extractor.push(observation)
        model = self._extractor.last_landmarks
        image_points = tuple((float(p[0]), float(p[1])) for p in points)
        image_points_smoothed = self._smooth_image_points(points[:, :2], timestamp_ms)
        model_points = None if model is None else tuple((float(p[0]), float(p[1])) for p in model)
        model_points_raw = tuple((float(p[0]), float(p[1])) for p in observation.landmarks)
        wrist_velocity = _as_vec2(self._extractor.last_wrist_velocity)
        wrist_acceleration = _as_vec2(self._extractor.last_wrist_acceleration)
        return HandSnapshot(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            hand_detected=True,
            handedness=observation.handedness,
            handedness_score=observation.handedness_score,
            detection_confidence=observation.detection_confidence,
            palm_scale=observation.palm_scale,
            image_points=image_points,
            model_points=model_points,
            model_points_raw=model_points_raw,
            landmark_count=len(image_points),
            inference_ms=inference_ms,
            smoothed=self._smoothing,
            wrist_velocity=wrist_velocity,
            wrist_acceleration=wrist_acceleration,
            image_points_smoothed=image_points_smoothed,
        )

    def _smooth_image_points(
        self, xy: npt.NDArray[np.float64], timestamp_ms: int
    ) -> tuple[Point2D, ...] | None:
        """One-Euro-smooth the image-space (x, y) for the live overlay (display only).

        Returns ``None`` when smoothing is disabled — the overlay then falls back to
        the raw detection. Never affects the model input or training data.
        """
        if self._image_smoother is None:
            return None
        smoothed = self._image_smoother.filter(xy, timestamp_ms)
        return tuple((float(p[0]), float(p[1])) for p in smoothed)

    @staticmethod
    def _primary_handedness(result: object) -> tuple[str, float]:
        handedness_list = getattr(result, "handedness", None)
        if not handedness_list or not handedness_list[0]:
            return "", 0.0
        top = handedness_list[0][0]
        return str(top.category_name), float(top.score)

    def _lost(self, timestamp_ms: int, frame_id: int, inference_ms: float) -> HandSnapshot:
        # Reset the extractor on tracking loss so smoothing never bridges the gap.
        self._extractor.reset()
        if self._image_smoother is not None:
            self._image_smoother.reset()
        return HandSnapshot(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            hand_detected=False,
            handedness="",
            handedness_score=0.0,
            detection_confidence=0.0,
            palm_scale=0.0,
            image_points=None,
            model_points=None,
            model_points_raw=None,
            landmark_count=0,
            inference_ms=inference_ms,
            smoothed=self._smoothing,
            wrist_velocity=None,
            wrist_acceleration=None,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            from mediapipe.tasks.python.vision import HandLandmarker

            assert isinstance(self._landmarker, HandLandmarker)
            self._landmarker.close()
            self._landmarker = None
            self._available = False
