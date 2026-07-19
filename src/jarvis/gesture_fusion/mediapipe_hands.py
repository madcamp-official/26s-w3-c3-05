"""MediaPipe Hand Landmarker 어댑터 → HandObservation.

`jarvis.gesture_fusion` 패키지에서 mediapipe를 직접 import하는 유일한 모듈이다 —
config/landmarks/(이후의) features·spotting·fusion은 순수 값(RawHandLandmarks·
HandObservation)만 다루므로 mediapipe나 모델 파일, 카메라 없이 단위 테스트할 수 있다
(pyproject.toml의 `vision` extra는 이 모듈에서만 필요하다).

이 어댑터는 `HandLandmarkSource` Protocol을 구현한다. 원시 랜드마크 추출만 여기서
하고, 손목 기준·손바닥 크기 정규화는 순수 `landmarks.normalize_hand`에 위임한다 —
그래서 정규화 규칙과 모델 소스를 독립적으로 바꿀 수 있다.

**입력 색상 규약(통합 시 주의)**: `process`의 `rgb_frame`은 반드시 **RGB** 채널
순서여야 한다(MediaPipe `SRGB`). OpenCV/웹캠 캡처는 기본이 BGR이므로, 프레임을
넘기기 전에 호출 측(캡처↔비전 배선 계층)에서 `cv2.cvtColor(frame, COLOR_BGR2RGB)`로
변환해야 한다 — Gaze의 `jarvis.gaze.cli`가 같은 규약을 쓴다. 색상이 뒤집힌 채로
넘기면 예외 없이 손 검출 품질만 조용히 떨어진다. 이 변환과 `capture.Frame` 언팩은
모듈 경계 규칙상 이 패키지가 아니라 앱/배선 계층의 책임이다(gesture_fusion은
runtime_protocol 내부 타입을 import하지 않는다).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from jarvis.gesture_fusion.config import (
    HAND_LANDMARK_COUNT,
    LANDMARK_DIMS,
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

        `rgb_frame`은 RGB 채널 순서여야 한다(모듈 docstring의 색상 규약 참조).
        `timestamp_ms`는 계약의 단일 monotonic clock 값을 그대로 넘겨야 하며,
        `detect_for_video`가 프레임 간 단조 증가를 요구한다(자체 시계로 재생성 금지).

        손을 못 찾으면 hand_detected=False인 관측값을 반환한다 — 추적 손실을
        지어낸 좌표로 감추지 않는다(development-principles.md 1·2절).
        """
        mp_image = MpImage(image_format=MpImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return _lost_tracking_observation(timestamp_ms, frame_id)

        best_index = self._select_primary_hand(result)
        landmarks = result.hand_landmarks[best_index]
        # z(깊이)는 단안 웹캠 추정값이라 노이즈가 커 버리고 x·y만 쓴다(config.LANDMARK_DIMS).
        points = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float64)
        if points.shape != (HAND_LANDMARK_COUNT, LANDMARK_DIMS):
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
        """선택한 손의 handedness 라벨과 score를 꺼낸다.

        handedness 정보가 없으면 score를 0.0으로 돌려준다 — 신호가 없을 때 최대
        신뢰도를 지어내지 않는다(development-principles.md 2절). 이 값은 검출
        신뢰도의 프록시로도 쓰이므로, 0.0이면 normalize_hand가 추적 손실로 처리한다.
        """
        handedness_list = getattr(result, "handedness", None)
        if not handedness_list or index >= len(handedness_list):
            return "", 0.0
        categories = handedness_list[index]
        if not categories:
            return "", 0.0
        top = categories[0]
        return str(top.category_name), float(top.score)
