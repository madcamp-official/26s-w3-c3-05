"""MediaPipe Face Landmarker adapter → FaceObservation.

README 7장 "입력 정보"의 원천이자 담당 범위의 "Face·iris landmark"·"head pose"
항목을 구현한다. `jarvis.gaze` 패키지에서 mediapipe를 직접 import하는 유일한
모듈이다 — features/smoothing/classifier/lock/engine은 순수 `FaceObservation`
값만 다루므로 mediapipe나 모델 파일, 카메라 없이 단위 테스트할 수 있다
(pyproject.toml의 `vision` extra는 이 모듈에서만 필요하다).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from jarvis.gaze.blink import AdaptiveBlinkDetector
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.features import FaceObservation, Vector3

FloatMatrix = npt.NDArray[np.float64]
RgbFrame = npt.NDArray[np.uint8]

try:
    from mediapipe import Image as MpImage
    from mediapipe import ImageFormat as MpImageFormat
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
    )
except ImportError as exc:  # pragma: no cover - only hit without the `vision` extra
    raise ImportError(
        "mediapipe is required for jarvis.gaze.landmarks; install with `pip install -e '.[vision]'`"
    ) from exc

# Canonical MediaPipe Face Landmarker indices (478-point mesh, iris included).
# 참고: https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
_LEFT_EYE_IRIS_CENTER = 473
_LEFT_EYE_INNER_CORNER = 362
_LEFT_EYE_OUTER_CORNER = 263
_LEFT_EYE_UPPER_LID = 386
_LEFT_EYE_LOWER_LID = 374

_RIGHT_EYE_IRIS_CENTER = 468
_RIGHT_EYE_INNER_CORNER = 133
_RIGHT_EYE_OUTER_CORNER = 33
_RIGHT_EYE_UPPER_LID = 159
_RIGHT_EYE_LOWER_LID = 145


def _iris_relative_position(
    landmarks: Any,
    iris_idx: int,
    inner_idx: int,
    outer_idx: int,
    upper_idx: int,
    lower_idx: int,
) -> tuple[float, float]:
    """홍채 중심을 눈 사각 영역 기준 -1..1로 정규화한다 (FaceObservation 좌표계)."""
    iris = landmarks[iris_idx]
    inner = landmarks[inner_idx]
    outer = landmarks[outer_idx]
    upper = landmarks[upper_idx]
    lower = landmarks[lower_idx]

    center_x = (inner.x + outer.x) / 2.0
    center_y = (upper.y + lower.y) / 2.0
    half_width = abs(outer.x - inner.x) / 2.0
    half_height = abs(lower.y - upper.y) / 2.0

    if half_width < 1e-6 or half_height < 1e-6:
        return 0.0, 0.0

    relative_x = (iris.x - center_x) / half_width
    relative_y = (center_y - iris.y) / half_height  # 이미지 좌표는 아래로 증가하므로 반전
    return (
        float(np.clip(relative_x, -1.0, 1.0)),
        float(np.clip(relative_y, -1.0, 1.0)),
    )


def _eye_center_normalized(
    landmarks: Any, inner_idx: int, outer_idx: int, upper_idx: int, lower_idx: int
) -> tuple[float, float]:
    """눈 네 경계 landmark의 중심을 이미지 정규화 좌표로 반환한다."""
    points = [landmarks[index] for index in (inner_idx, outer_idx, upper_idx, lower_idx)]
    return (
        float(np.clip(sum(point.x for point in points) / len(points), 0.0, 1.0)),
        float(np.clip(sum(point.y for point in points) / len(points), 0.0, 1.0)),
    )


def _eye_open_ratio(
    landmarks: Any, inner_idx: int, outer_idx: int, upper_idx: int, lower_idx: int
) -> float:
    """Return eyelid opening normalized by eye width."""
    inner = landmarks[inner_idx]
    outer = landmarks[outer_idx]
    upper = landmarks[upper_idx]
    lower = landmarks[lower_idx]
    width = math.hypot(outer.x - inner.x, outer.y - inner.y)
    if width < 1e-6:
        return 0.0
    return float(abs(lower.y - upper.y) / width)


def rotation_matrix_to_euler_deg(matrix: FloatMatrix) -> tuple[float, float, float]:
    """회전 행렬(3x3 또는 4x4의 좌상단)에서 (yaw, pitch, roll)을 degree로 추출한다.

    MediaPipe facial transformation matrix의 좌표계(오른손, x-우측·y-상단·
    z-사용자쪽)를 가정한 표준 분해식이다. 실 카메라로 첫 통합 테스트를 할 때
    (README 16장 Day 1) 부호나 축이 뒤바뀌어 보이면 이 함수만 조정하면 된다 —
    calibration·target classifier는 등록·실사용에 동일한 변환을 일관되게 쓰는 한
    절대적인 부호 규약에 의존하지 않는다.
    """
    r = matrix[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    # MediaPipe transformation matrix의 x축 회전은 사용자가 위를 볼 때 음수다.
    # FaceObservation 계약은 pitch 양수를 화면 위쪽으로 정의하므로 여기서 반전한다.
    return math.degrees(yaw), -math.degrees(pitch), math.degrees(roll)


def translation_from_transform(matrix: FloatMatrix) -> Vector3:
    """4x4 변환 행렬에서 머리(얼굴 모델 원점)의 카메라 기준 위치를 추출한다.

    `matrix[:3, 3]`(마지막 열의 위쪽 3개 성분)이 표준 동차좌표 변환의 translation
    성분이다. MediaPipe의 표준 얼굴 모델 크기 가정에 기반한 근사 스케일이며 실측
    검증된 계량값이 아니다(models/README.md 참고) — 3D 삼각측량의 원점으로만
    쓰이고, 각도 기반 경로(rotation_matrix_to_euler_deg)에는 영향을 주지 않는다.
    """
    return np.array(matrix[:3, 3], dtype=np.float64)


def _lost_tracking_observation(timestamp_ms: int, frame_id: int) -> FaceObservation:
    """얼굴을 찾지 못한 프레임의 관측값 — 추적 손실을 지어낸 값으로 감추지 않는다."""
    return FaceObservation(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        left_iris_relative=(0.0, 0.0),
        right_iris_relative=(0.0, 0.0),
        head_yaw_deg=0.0,
        head_pitch_deg=0.0,
        head_roll_deg=0.0,
        eye_tracking_confidence=0.0,
        face_tracking_confidence=0.0,
        face_detected=False,
        eyes_open=False,
    )


class FaceLandmarkerAdapter:
    """MediaPipe Face Landmarker를 감싸 프레임마다 FaceObservation을 만든다."""

    def __init__(
        self,
        model_asset_path: str | Path,
        num_faces: int = 1,
        config: GazeConfig = GazeConfig(),
    ) -> None:
        model_path = Path(model_asset_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face Landmarker model asset not found at {model_path}. "
                "models/README.md에 기록된 확보 방법을 따라 내려받은 뒤 경로를 지정하라."
            )
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=num_faces,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._blink_detector = AdaptiveBlinkDetector(config)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarkerAdapter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def process(self, rgb_frame: RgbFrame, timestamp_ms: int, frame_id: int) -> FaceObservation:
        """RGB 프레임 하나를 처리해 FaceObservation을 만든다.

        얼굴을 못 찾으면 face_detected=False인 관측값을 반환한다 — 추적 손실을
        지어낸 값으로 감추지 않는다(development-principles.md 1·2절).
        """
        mp_image = MpImage(image_format=MpImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks or not result.facial_transformation_matrixes:
            return _lost_tracking_observation(timestamp_ms, frame_id)

        landmarks = result.face_landmarks[0]
        transform: FloatMatrix = np.array(
            result.facial_transformation_matrixes[0], dtype=np.float64
        )
        yaw_deg, pitch_deg, roll_deg = rotation_matrix_to_euler_deg(transform)
        head_position_mm = translation_from_transform(transform)

        left_iris = _iris_relative_position(
            landmarks,
            _LEFT_EYE_IRIS_CENTER,
            _LEFT_EYE_INNER_CORNER,
            _LEFT_EYE_OUTER_CORNER,
            _LEFT_EYE_UPPER_LID,
            _LEFT_EYE_LOWER_LID,
        )
        right_iris = _iris_relative_position(
            landmarks,
            _RIGHT_EYE_IRIS_CENTER,
            _RIGHT_EYE_INNER_CORNER,
            _RIGHT_EYE_OUTER_CORNER,
            _RIGHT_EYE_UPPER_LID,
            _RIGHT_EYE_LOWER_LID,
        )
        left_eye_center = _eye_center_normalized(
            landmarks,
            _LEFT_EYE_INNER_CORNER,
            _LEFT_EYE_OUTER_CORNER,
            _LEFT_EYE_UPPER_LID,
            _LEFT_EYE_LOWER_LID,
        )
        right_eye_center = _eye_center_normalized(
            landmarks,
            _RIGHT_EYE_INNER_CORNER,
            _RIGHT_EYE_OUTER_CORNER,
            _RIGHT_EYE_UPPER_LID,
            _RIGHT_EYE_LOWER_LID,
        )
        left_eye_open_ratio = _eye_open_ratio(
            landmarks,
            _LEFT_EYE_INNER_CORNER,
            _LEFT_EYE_OUTER_CORNER,
            _LEFT_EYE_UPPER_LID,
            _LEFT_EYE_LOWER_LID,
        )
        right_eye_open_ratio = _eye_open_ratio(
            landmarks,
            _RIGHT_EYE_INNER_CORNER,
            _RIGHT_EYE_OUTER_CORNER,
            _RIGHT_EYE_UPPER_LID,
            _RIGHT_EYE_LOWER_LID,
        )

        # Face Landmarker Tasks API는 얼굴 전체에 대한 단일 confidence 스칼라를
        # 내주지 않는다. 얼굴이 검출된 프레임은 1.0으로 다루고, 검출 실패(위에서
        # 이미 처리됨)만 0.0으로 구분한다 — 실제로 중요한 신호인 "추적 손실 여부"는
        # 이 이진 근사로 충분히 표현된다.
        confidence = 1.0

        eyes_open = self._blink_detector.update(
            left_eye_open_ratio,
            right_eye_open_ratio,
        )
        left_eye_baseline, right_eye_baseline = self._blink_detector.eye_baselines

        return FaceObservation(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            left_iris_relative=left_iris,
            right_iris_relative=right_iris,
            head_yaw_deg=yaw_deg,
            head_pitch_deg=pitch_deg,
            head_roll_deg=roll_deg,
            eye_tracking_confidence=confidence,
            face_tracking_confidence=confidence,
            face_detected=True,
            left_eye_center_normalized=left_eye_center,
            right_eye_center_normalized=right_eye_center,
            eyes_open=eyes_open,
            left_eye_open_ratio=left_eye_open_ratio,
            right_eye_open_ratio=right_eye_open_ratio,
            left_eye_open_baseline=left_eye_baseline,
            right_eye_open_baseline=right_eye_baseline,
            head_position_mm=head_position_mm,
        )
