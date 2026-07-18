"""MediaPipe Hand Landmarker 어댑터 → HandObservation.

`jarvis.gesture_fusion` 패키지에서 mediapipe를 직접 import하는 유일한 모듈이다 —
config/landmarks/(이후의) features·spotting·fusion은 순수 값(RawHandLandmarks·
HandObservation)만 다루므로 mediapipe나 모델 파일, 카메라 없이 단위 테스트할 수 있다
(pyproject.toml의 `vision` extra는 이 모듈에서만 필요하다).

이 어댑터는 `HandLandmarkSource` Protocol을 구현한다. 원시 랜드마크 추출만 여기서
하고, 손목 기준·손바닥 크기 정규화는 순수 `landmarks.normalize_hand`에 위임한다 —
그래서 정규화 규칙과 모델 소스를 독립적으로 바꿀 수 있다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from jarvis.gesture_fusion.config import (
    HAND_LANDMARK_COUNT,
    DEFAULT_GESTURE_CONFIG,
    GestureConfig,
)
from jarvis.gesture_fusion.landmarks import (
    HandObservation,
    RawHandLandmarks,
    _lost_tracking_observation,
    normalize_hand,
)

RgbFrame = npt.NDArray[np.uint8]

try:
    from mediapipe import Image as MpImage
    from mediapipe import ImageFormat as MpImageFormat
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )
except ImportError as exc:  # pragma: no cover - only hit without the `vision` extra
    raise ImportError(
        "mediapipe is required for jarvis.gesture_fusion.mediapipe_hands; install with "
        "`pip install -e '.[vision]'`"
    ) from exc


class MediaPipeHandLandmarker:
    """MediaPipe Hand Landmarker를 감싸 프레임마다 HandObservation을 만든다.

    `HandLandmarkSource` Protocol의 MVP 구현이다. 여러 손이 잡히면
    handedness/detection score가 가장 높은 손 하나만 골라 downstream으로 넘긴다
    (MVP는 주 조작 손 1개; config.num_hands로 검출 상한을 조절한다).
    """

    def __init__(
        self,
        model_asset_path: str | Path,
        config: GestureConfig = DEFAULT_GESTURE_CONFIG,
    ) -> None:
        model_path = Path(model_asset_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Hand Landmarker model asset not found at {model_path}. "
                "models/README.md에 기록된 확보 방법을 따라 내려받은 뒤 경로를 지정하라."
            )
        self._config = config
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=config.num_hands,
            min_hand_detection_confidence=config.min_hand_detection_confidence,
            min_hand_presence_confidence=config.min_hand_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "MediaPipeHandLandmarker":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def process(self, rgb_frame: RgbFrame, timestamp_ms: int, frame_id: int) -> HandObservation:
        """RGB 프레임 하나를 처리해 정규화된 HandObservation을 반환한다.

        손을 못 찾으면 hand_detected=False인 관측값을 반환한다 — 추적 손실을
        지어낸 좌표로 감추지 않는다(development-principles.md 1·2절).
        """
        mp_image = MpImage(image_format=MpImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return _lost_tracking_observation(timestamp_ms, frame_id)

        best_index = self._select_primary_hand(result)
        landmarks = result.hand_landmarks[best_index]
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
        if points.shape != (HAND_LANDMARK_COUNT, 3):
            # 모델이 21점을 채우지 못한 비정상 프레임은 추적 손실로 처리한다.
            return _lost_tracking_observation(timestamp_ms, frame_id)

        handedness, handedness_score = self._primary_handedness(result, best_index)

        # MediaPipe HandLandmarkerResult는 손별 검출 score를 공개 API로 내주지 않는다.
        # 반환된 손은 이미 모델 내부의 min_hand_detection_confidence를 통과했으므로,
        # 검출 신뢰도의 프록시로 handedness score를 재사용한다(원격 소스는 둘을 독립적으로
        # 보고할 수 있어 RawHandLandmarks는 두 필드를 분리해 둔다).
        raw = RawHandLandmarks(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            points=points,
            handedness=handedness,
            detection_confidence=handedness_score,
            handedness_score=handedness_score,
        )
        return normalize_hand(raw, self._config)

    @staticmethod
    def _select_primary_hand(result: object) -> int:
        """검출 score가 가장 높은 손의 인덱스를 고른다 (여러 손 중 주 조작 손)."""
        handedness_list = getattr(result, "handedness", None)
        if not handedness_list:
            return 0
        best_index = 0
        best_score = -1.0
        for index, categories in enumerate(handedness_list):
            score = categories[0].score if categories else 0.0
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    @staticmethod
    def _primary_handedness(result: object, index: int) -> tuple[str, float]:
        """선택한 손의 handedness 라벨과 score를 꺼낸다."""
        handedness_list = getattr(result, "handedness", None)
        if not handedness_list or index >= len(handedness_list):
            return "", 1.0
        categories = handedness_list[index]
        if not categories:
            return "", 1.0
        top = categories[0]
        return str(top.category_name), float(top.score)
