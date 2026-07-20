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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from jarvis.gesture_fusion.config import DEFAULT_GESTURE_CONFIG, LANDMARK_DIMS, GestureConfig
from jarvis.gesture_fusion.features import HandFeatureExtractor
from jarvis.gesture_fusion.landmarks import (
    RawHandLandmarks,
    is_palm_tilted,
    normalize_hand,
    palm_tilt_degrees,
)
from jarvis.gesture_fusion.pose_protocol import NullPoseClassifier, PoseClassifier, PosePrediction
from jarvis.gesture_fusion.smoothing import OneEuroFilter

Point2D = tuple[float, float]


def _as_vec2(vec: npt.NDArray[np.float64] | None) -> tuple[float, float] | None:
    """Convert a length-2 array to a plain float tuple for the UI (None passes through)."""
    if vec is None:
        return None
    return (float(vec[0]), float(vec[1]))


@dataclass(frozen=True, slots=True)
class ImageSmoothingConfig:
    """웹캠 오버레이(이미지 좌표) 전용 One-Euro 파라미터 — 표시 전용.

    `GestureConfig.smoothing_*`(모델 입력용)와 **값을 공유하지 않는다.** 두 필터가
    서로 다른 좌표계를 다루기 때문이다:

    - 모델 입력: 손바닥 크기로 정규화된 좌표(palm-width 단위). 손 이동 속도가 보통 초당 1~5.
    - 이 필터: 이미지 정규화 좌표([0, 1] 프레임 비율). 같은 손 이동이 초당 0.2~1.0 정도로
      **수치가 약 5배 작다**(palm_scale이 프레임 폭의 15~25% 수준이므로).

    One-Euro의 적응 컷오프는 `min_cutoff + beta × |속도|`이고 속도가 신호와 같은 단위로
    들어간다. 그래서 palm 공간용 `beta`(0.5)를 이미지 공간에 그대로 쓰면 컷오프가 거의
    열리지 않아 사실상 고정 ~1Hz 저역통과가 되고, 30fps에서 시정수 약 170ms의 지연이
    생긴다 — 스켈레톤이 손을 눈에 띄게 늦게 따라오던 원인이다. 여기서는 이미지 단위에
    맞춘 `beta`를 따로 둔다.

    `pointer.hand_cursor`가 같은 이유로 `ref_*`를 분리해 둔 것과 같은 판단이며, 표시
    감도를 모델 재현성 제약(학습 데이터와 동일 전처리)에 묶지 않는 효과도 같다.
    """

    min_cutoff: float = 1.5
    """정지한 손의 지터를 잡는 기본 강도(Hz). 낮을수록 정지 시 부드럽지만 지연이 커진다."""

    beta: float = 20.0
    """속도에 따른 컷오프 개방 계수 — **이미지 좌표 단위** 기준. 지연을 좌우하는 주 레버다.

    20이면 느린 이동(0.2/s)에서 컷오프가 5.5Hz까지 열려 30fps 시정수가 약 60ms로
    떨어지고, 빠른 이동에서는 사실상 통과된다. 반면 정지 상태(≈0.01/s)에서는 1.7Hz에
    머물러 지터 억제는 그대로 유지된다."""

    d_cutoff: float = 1.5
    """내부 속도 추정의 평활 컷오프(Hz). 낮추면 이동 시작 순간 컷오프가 늦게 열린다."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_cutoff) or self.min_cutoff <= 0.0:
            raise ValueError("min_cutoff must be finite and positive")
        if not math.isfinite(self.d_cutoff) or self.d_cutoff <= 0.0:
            raise ValueError("d_cutoff must be finite and positive")
        if not math.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("beta must be finite and non-negative")


DEFAULT_IMAGE_SMOOTHING = ImageSmoothingConfig()


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
    # 손바닥 축이 이미지 평면과 이루는 각(도). z에서만 구할 수 있어 소스가 계산한다.
    # None = 알 수 없음(게이트를 걸지 않는다). ``palm_tilted``는 이 값이 설정된 임계를
    # 넘어 자세 판정이 거부되는 상태 — 조용히 무시하지 않고 화면에 드러내기 위한 필드다.
    palm_tilt_degrees: float | None = None
    palm_tilted: bool = False
    # 정적 자세 판정. 모델이 없으면 `trusted=False`에 사유가 담긴다 — 자세를 지어내지
    # 않으며, 학습 안 한 상태가 "인식이 안 된다"로 오해되지 않게 UI에 그대로 드러낸다.
    pose: PosePrediction | None = None


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


def _load_pose_classifier(path: Path | None, config: GestureConfig) -> PoseClassifier:
    """자세 분류기를 싣되, 실패해도 프로브 전체를 죽이지 않는다.

    모델이 없거나 전처리가 어긋나면 `NullPoseClassifier`로 대체하고 그 사유를 판정
    결과에 담는다 — 손 추적 자체는 계속 보여야 원인을 진단할 수 있다.
    """
    if path is None:
        return NullPoseClassifier()
    try:
        from jarvis.gesture_fusion.pose_classifier import TorchPoseClassifier

        return TorchPoseClassifier(path, config)
    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 사유를 그대로 노출한다
        return NullPoseClassifier(reason=f"자세 모델 로드 실패: {exc}")


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
        image_smoothing: ImageSmoothingConfig = DEFAULT_IMAGE_SMOOTHING,
        pose_model_path: Path | None = None,
    ) -> None:
        self._model_path = model_path
        self._pose_classifier: PoseClassifier = _load_pose_classifier(pose_model_path, config)
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
        # Its parameters come from ``ImageSmoothingConfig``, NOT from ``config`` —
        # the two filters work in different coordinate spaces, and reusing the
        # palm-space beta here made the cutoff barely open (≈170 ms of visible lag).
        self._image_smoothing = image_smoothing
        self._image_smoother: OneEuroFilter | None = (
            OneEuroFilter(
                min_cutoff=image_smoothing.min_cutoff,
                beta=image_smoothing.beta,
                d_cutoff=image_smoothing.d_cutoff,
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
        # 기울기만 z에서 계산해 넘긴다(좌표는 2D 유지) — 화면 밖 회전은 z에만 있다.
        tilt = palm_tilt_degrees(
            np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64), self._config
        )

        handedness, score = self._primary_handedness(result)
        raw = RawHandLandmarks(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            points=points,
            handedness=handedness,
            detection_confidence=score,
            handedness_score=score,
            palm_tilt_degrees=tilt,
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
            palm_tilt_degrees=observation.palm_tilt_degrees,
            palm_tilted=is_palm_tilted(observation, self._config),
            pose=self._classify_pose(model, observation.palm_tilt_degrees),
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

    def _classify_pose(
        self, model_points: object, tilt: float | None
    ) -> PosePrediction:
        """평활된 모델 입력 좌표로 자세를 판정한다 — 학습과 동일한 값을 넣는다.

        `model_points`가 없으면(첫 프레임 등) 지어내지 않고 거부 사유를 남긴다.
        """
        if model_points is None:
            return PosePrediction(
                label="", confidence=0.0, trusted=False,
                reason="모델 입력 준비 전", palm_tilt_degrees=tilt,
            )
        return self._pose_classifier.classify(np.asarray(model_points), tilt)

    def close(self) -> None:
        if self._landmarker is not None:
            from mediapipe.tasks.python.vision import HandLandmarker

            assert isinstance(self._landmarker, HandLandmarker)
            self._landmarker.close()
            self._landmarker = None
            self._available = False
