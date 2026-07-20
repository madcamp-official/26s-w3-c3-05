"""참고 레포 방식 랜드마크 추출 어댑터 — 레거시 `mp.solutions.hands`.

Kazuhito00/hand-gesture-recognition-using-mediapipe의 `app.py`가 쓰는 검출 경로를
그대로 옮긴 것이다:

    RGB 프레임 → `hands.process(rgb)` → `multi_hand_landmarks[0]` → 21점 정규화 좌표

`mediapipe_hands.MediaPipeHandLandmarker`(Tasks API)와 **같은 `HandLandmarkSource`
Protocol**을 구현하므로 downstream(features·TCN·fusion)은 어느 쪽을 쓰든 바뀌지
않는다. 두 어댑터의 차이는 검출 엔진뿐이고, 손목 기준·손바닥 크기 정규화는 둘 다
순수 `landmarks.normalize_hand`에 위임한다.

Tasks API와 다른 점(이식하며 알아둘 것)
--------------------------------------
- **모델 파일이 필요 없다.** Solutions는 그래프와 가중치를 mediapipe 휠에 내장하고
  있어 `hand_landmarker.task` 없이 돈다. Tasks API 어댑터가 요구하는
  `model_asset_path`가 여기엔 없다.
- **타임스탬프를 받지 않는다.** Solutions는 내부적으로 프레임 순서를 스스로
  관리한다(Tasks API의 `detect_for_video`는 단조 증가 타임스탬프를 요구했다).
  Protocol 시그니처를 맞추기 위해 `timestamp_ms`는 받되 관측값에 실어 보내기만 한다.
- **21점 토폴로지는 동일**하므로 정규화·feature 경로는 그대로 재사용된다.
- **handedness 라벨이 Tasks API와 반대로 나온다.** 참고 레포 `app.py`는 검출 *전에*
  `cv.flip(frame, 1)`로 좌우를 뒤집고 그 거울상을 엔진에 넣기 때문에, 같은 손을
  두고 Solutions는 "Right", Tasks는 "Left"라고 답한다(실측 확인). 이 어댑터는
  참고 레포와 달리 **반전하지 않은 원본 프레임**을 받는 규약이라(표시용 반전은
  UI 계층 책임 — `mediapipe_hands`와 동일), 라벨은 엔진이 준 값을 그대로 싣는다.
  handedness에 의존하는 downstream이 있다면 백엔드를 바꿀 때 이 반전을 반드시
  고려해야 한다 — 랜드마크 좌표는 거의 같아도 라벨은 뒤집힌다.

이 모듈은 `solutions`가 살아있는 mediapipe에서만 import된다(0.10.15+ slim 패키징은
`mediapipe.solutions`를 뺐다). pyproject의 `vision` extra가 그 조건을 만족하는
버전으로 고정돼 있다.

**입력 색상 규약**: `mediapipe_hands`와 동일하게 `rgb_frame`은 RGB 채널 순서여야
한다. BGR을 넘기면 예외 없이 검출 품질만 조용히 떨어진다.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - only hit without the `vision` extra
    raise ImportError(
        "mediapipe is required for jarvis.gesture_fusion.solutions_hands; install with "
        "`pip install -e '.[vision]'`"
    ) from exc

if not hasattr(mp, "solutions"):  # pragma: no cover - depends on installed wheel
    raise ImportError(
        f"설치된 mediapipe {mp.__version__}에 레거시 solutions API가 없습니다. "
        "참고 레포 방식 검출에는 solutions가 포함된 버전이 필요합니다 "
        "(pyproject의 vision extra 핀을 확인하세요)."
    )


@dataclass(frozen=True, slots=True)
class SolutionsHandDetection:
    """레거시 엔진이 한 프레임에서 뽑은 원시 검출 결과(이미지 좌표계).

    좌표는 이미지 정규화 [0, 1]이라 해상도와 무관하다 — 웹캠 오버레이에 그리려면
    프레임 크기를 곱하면 된다. 정규화(손목 원점·palm_scale)를 거치기 **전** 값이다.
    """

    points: npt.NDArray[np.float64]  # (21, 2) 이미지 좌표 [0, 1]
    handedness: str
    handedness_score: float


class SolutionsHandDetector:
    """참고 레포와 동일한 설정의 `mp.solutions.hands.Hands` 래퍼(원시 검출 전용).

    이미지 좌표를 그대로 돌려주므로 웹캠 위에 스켈레톤을 그리는 디버그 툴이 직접 쓸
    수 있다. 정규화된 관측값이 필요하면 `SolutionsHandLandmarker`를 쓴다.
    """

    def __init__(self, config: GestureConfig = DEFAULT_GESTURE_CONFIG) -> None:
        self._config = config
        self._hands = mp.solutions.hands.Hands(
            # 참고 app.py 기본값: 정지 이미지 모드 off(비디오 추적), 손 1개 우선.
            static_image_mode=False,
            max_num_hands=config.num_hands,
            min_detection_confidence=config.min_hand_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

    def detect(self, rgb_frame: RgbFrame) -> SolutionsHandDetection | None:
        """RGB 프레임 하나에서 주 조작 손을 검출한다. 손이 없으면 None.

        참고 레포와 동일하게 **첫 번째 손**만 쓴다. Solutions는 검출 score를 손별로
        공개하지 않으므로 handedness score를 신뢰도 프록시로 쓴다(Tasks API 어댑터와
        같은 처리 — `mediapipe_hands`의 주석 참조).
        """
        result = self._hands.process(rgb_frame)
        if not result.multi_hand_landmarks:
            return None

        landmarks = result.multi_hand_landmarks[0]
        # z(깊이)는 단안 웹캠 추정값이라 노이즈가 커 버리고 x·y만 쓴다.
        points = np.array([[lm.x, lm.y] for lm in landmarks.landmark], dtype=np.float64)
        if points.shape != (HAND_LANDMARK_COUNT, LANDMARK_DIMS):
            return None

        handedness = ""
        score = 0.0
        if result.multi_handedness:
            classification = result.multi_handedness[0].classification[0]
            handedness = str(classification.label)
            score = float(classification.score)
        return SolutionsHandDetection(
            points=points, handedness=handedness, handedness_score=score
        )

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "SolutionsHandDetector":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class SolutionsHandLandmarker:
    """참고 레포 검출 경로를 쓰는 `HandLandmarkSource` 구현.

    Tasks API 어댑터(`MediaPipeHandLandmarker`)의 형제 구현이며, downstream 계약은
    완전히 동일하다 — 손을 못 찾으면 `hand_detected=False`인 관측값을 돌려주고
    추적 손실을 지어낸 좌표로 감추지 않는다.
    """

    def __init__(self, config: GestureConfig = DEFAULT_GESTURE_CONFIG) -> None:
        self._config = config
        self._detector = SolutionsHandDetector(config)

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "SolutionsHandLandmarker":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def process(self, rgb_frame: RgbFrame, timestamp_ms: int, frame_id: int) -> HandObservation:
        """RGB 프레임 하나를 처리해 정규화된 HandObservation을 반환한다.

        `timestamp_ms`는 관측값에 실어 보내기만 한다 — Solutions 엔진은 Tasks API와
        달리 타임스탬프를 요구하지 않고 프레임 순서를 내부에서 관리한다.
        """
        detection = self._detector.detect(rgb_frame)
        if detection is None:
            return _lost_tracking_observation(timestamp_ms, frame_id)

        raw = RawHandLandmarks(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            points=detection.points,
            handedness=detection.handedness,
            detection_confidence=detection.handedness_score,
            handedness_score=detection.handedness_score,
        )
        return normalize_hand(raw, self._config)
