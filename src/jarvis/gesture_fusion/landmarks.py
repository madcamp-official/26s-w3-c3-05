"""Hand landmark 정규화 — 원시 21점 랜드마크 → HandObservation.

README 8장 처리 과정의 앞 두 단계를 구현한다:

    MediaPipe Hand Landmark → 손목 기준 좌표 정규화 → 손바닥 크기 정규화

이 모듈은 **mediapipe에 의존하지 않는다**. 실제 모델 어댑터(`mediapipe_hands.py`)나
원격 GPU 서버 소스가 원시 랜드마크(`RawHandLandmarks`)만 넘겨주면, 정규화는 전부
여기 순수 함수에서 일어난다 — 그래서 정규화 로직은 카메라·모델 없이 단위 테스트할 수
있고, landmark 소스는 `HandLandmarkSource` Protocol로 자유롭게 교체된다
(2026-07-18 결정: 추론 위치를 교체 가능한 경계로 분리).

정규화는 평행이동(손목 기준)과 스케일(손바닥 크기)만 제거한다. **회전은 정규화하지
않는다** — README 8장의 "볼륨 조절: 손목 회전" 제스처가 회전 신호에 의존하므로,
회전을 없애면 그 제스처를 판별할 수 없다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from jarvis.gesture_fusion.config import (
    HAND_LANDMARK_COUNT,
    LANDMARK_DIMS,
    DEFAULT_GESTURE_CONFIG,
    GestureConfig,
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RawHandLandmarks:
    """landmark 소스가 내는 원시(정규화 전) 손 랜드마크.

    `points`는 (21, 2) 배열로, MediaPipe Hand Landmarker의 이미지 정규화 좌표
    (x, y ∈ 대략 [0, 1])를 그대로 담는다. z(깊이)는 단안 웹캠 추정값이라 노이즈가
    커 사용하지 않는다(config.LANDMARK_DIMS). 좌표계 변환·스케일 정규화는 이 값이
    아니라 `normalize_hand`에서 수행한다.

    `handedness`는 소스가 보고한 "Left"/"Right" 문자열이다(셀피 미러 뷰에서는
    좌우가 뒤집혀 보일 수 있으므로 소스 보고값을 가공 없이 보존한다).

    `detection_confidence`와 `handedness_score`는 **서로 다른 값**이다.
    `detection_confidence`는 "이게 손이 맞다"는 소스의 검출 신뢰도(추적 손실 게이트에
    쓰인다)이고, `handedness_score`는 "Left/Right 판별"의 분류 신뢰도다. 원격 소스는
    둘을 독립적으로 보고할 수 있고, MediaPipe는 손별 검출 score를 공개 API로 내주지
    않아 어댑터가 handedness score를 검출 신뢰도의 프록시로 재사용한다(그 사정은
    `mediapipe_hands.py`에 기록).
    """

    timestamp_ms: int
    frame_id: int
    points: FloatArray
    handedness: str
    detection_confidence: float
    handedness_score: float
    # 손바닥 축이 이미지 평면과 이루는 각(도). z가 있어야 계산되므로 **소스가 채운다**
    # — 여기 `points`는 2D라 이 값을 되계산할 수 없다. z를 좌표로 쓰지 않는다는 결정
    # (LANDMARK_DIMS=2)은 그대로 두고, 화면 밖 회전이라는 z만 아는 정보 하나만 각도로
    # 요약해 넘긴다. z를 못 내는 소스는 None을 두며, 그때 게이트는 적용되지 않는다.
    palm_tilt_degrees: float | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0 or self.frame_id < 0:
            raise ValueError("timestamp_ms and frame_id must be non-negative")
        if self.palm_tilt_degrees is not None and (
            not math.isfinite(self.palm_tilt_degrees) or not 0.0 <= self.palm_tilt_degrees <= 90.0
        ):
            raise ValueError("palm_tilt_degrees must be finite and within [0, 90]")
        if self.points.shape != (HAND_LANDMARK_COUNT, LANDMARK_DIMS):
            raise ValueError(
                f"points must have shape ({HAND_LANDMARK_COUNT}, {LANDMARK_DIMS}), got {self.points.shape}"
            )
        if not np.all(np.isfinite(self.points)):
            raise ValueError("raw landmark points must all be finite")
        if not math.isfinite(self.detection_confidence) or not 0.0 <= self.detection_confidence <= 1.0:
            raise ValueError("detection_confidence must be finite and within [0, 1]")
        if not math.isfinite(self.handedness_score) or not 0.0 <= self.handedness_score <= 1.0:
            raise ValueError("handedness_score must be finite and within [0, 1]")


@dataclass(frozen=True, slots=True)
class HandObservation:
    """단일 프레임의 정규화된 손 관측값 (README 8장 처리 과정 2·3단계 완료 상태).

    좌표계: `landmarks`는 (21, 2) 배열로, `config.origin_index`(기본: 손목)를
    원점으로 옮기고 손바닥 크기(config의 root→tip 거리)로 나눠 스케일을 제거한
    좌표다. 따라서 카메라와의 거리·프레임 내 위치와 무관하게 같은 손 모양이면 거의
    같은 값이 나온다. 회전은 보존된다(손목 회전 제스처용).

    `wrist_position`은 (2,) 배열로, 원점화하지 **않고** palm_scale로만 나눈 손목 좌표다
    — `landmarks`가 매 프레임 손목을 (0,0,0)으로 지워 잃어버리는 손 전체의 평행이동
    신호를 보존한다. palm_scale로 정규화해 카메라 거리에는 독립이다. feature 계층이
    이를 causal 차분해 손목 이동 속도·가속도(swipe 판별 신호)를 만든다
    (documents/decisions.md 2026-07-19). 추적 손실 프레임에서는 0벡터다.

    `detection_confidence`/`handedness_score`의 의미는 `RawHandLandmarks`와 같다
    (전자는 검출 신뢰도, 후자는 Left/Right 분류 신뢰도). 소스가 보고한 값을
    가공 없이 전파한다.

    `hand_detected=False`이면 추적 손실 프레임이다 — 지어낸 좌표가 아니라 0으로
    채운 값이며, downstream은 이를 실행이 아니라 거부/대기로 다뤄야 한다
    (development-principles.md 2절).
    """

    timestamp_ms: int
    frame_id: int
    landmarks: FloatArray
    handedness: str
    palm_scale: float
    detection_confidence: float
    handedness_score: float
    hand_detected: bool
    # palm_scale로만 정규화한(원점화 없는) 손목 좌표. 기본값 0벡터 = 평행이동 신호 없음.
    # 실사용 경로(normalize_hand)는 항상 실제 값을 채운다.
    wrist_position: FloatArray = field(
        default_factory=lambda: np.zeros(LANDMARK_DIMS, dtype=np.float64)
    )
    # 소스가 보고한 손바닥 기울기(도). None이면 소스가 z를 못 내 게이트를 걸 수 없다.
    palm_tilt_degrees: float | None = None

    @property
    def palm_tilted(self) -> bool:
        """이 관측값이 기울기 게이트에 걸리는가 — `DEFAULT_GESTURE_CONFIG` 기준.

        다른 임계로 판단하려면 `is_palm_tilted(observation, config)`를 쓴다.
        """
        return is_palm_tilted(self, DEFAULT_GESTURE_CONFIG)

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0 or self.frame_id < 0:
            raise ValueError("timestamp_ms and frame_id must be non-negative")
        if self.landmarks.shape != (HAND_LANDMARK_COUNT, LANDMARK_DIMS):
            raise ValueError(
                f"landmarks must have shape ({HAND_LANDMARK_COUNT}, {LANDMARK_DIMS}), got {self.landmarks.shape}"
            )
        if not np.all(np.isfinite(self.landmarks)):
            raise ValueError("normalized landmarks must all be finite")
        if self.wrist_position.shape != (LANDMARK_DIMS,):
            raise ValueError(
                f"wrist_position must have shape ({LANDMARK_DIMS},), got {self.wrist_position.shape}"
            )
        if not np.all(np.isfinite(self.wrist_position)):
            raise ValueError("wrist_position must be finite")
        if not math.isfinite(self.palm_scale) or self.palm_scale < 0.0:
            raise ValueError("palm_scale must be finite and non-negative")
        if not math.isfinite(self.detection_confidence) or not 0.0 <= self.detection_confidence <= 1.0:
            raise ValueError("detection_confidence must be finite and within [0, 1]")
        if not math.isfinite(self.handedness_score) or not 0.0 <= self.handedness_score <= 1.0:
            raise ValueError("handedness_score must be finite and within [0, 1]")
        if self.palm_tilt_degrees is not None and (
            not math.isfinite(self.palm_tilt_degrees) or not 0.0 <= self.palm_tilt_degrees <= 90.0
        ):
            raise ValueError("palm_tilt_degrees must be finite and within [0, 90]")


def palm_tilt_degrees(
    points_3d: FloatArray,
    config: GestureConfig = DEFAULT_GESTURE_CONFIG,
) -> float | None:
    """손바닥 축이 이미지 평면과 이루는 각(도)을 (x, y, z) 랜드마크에서 계산한다.

    손바닥 축 = `palm_scale_root_index` → `palm_scale_tip_index`(기본: 손목→중지 MCP).
    2D 길이 / 3D 길이 = cos(기울기각)이므로 arccos로 각을 얻는다. 0°는 손바닥 축이
    화면과 나란한 상태, 90°는 카메라를 정면으로 겨눈 상태다.

    **z를 좌표로 쓰는 것이 아니다.** z는 노이즈가 커 좌표에서 뺐지만(LANDMARK_DIMS=2),
    화면 밖 회전은 원리적으로 z에만 있는 정보다. 여기서는 각도 하나로 요약해 굵은 임계
    판정에만 쓰므로, 좌표로 쓸 때만큼 노이즈에 민감하지 않다.

    퇴화한 입력(길이 0)에서는 각을 정의할 수 없어 `None`을 반환한다 — 0°로 지어내면
    기울어진 손이 게이트를 통과해 버린다.
    """
    if points_3d.shape != (HAND_LANDMARK_COUNT, 3) or not np.all(np.isfinite(points_3d)):
        return None
    axis = points_3d[config.palm_scale_tip_index] - points_3d[config.palm_scale_root_index]
    length_3d = float(np.linalg.norm(axis))
    length_2d = float(np.linalg.norm(axis[:2]))
    if not math.isfinite(length_3d) or length_3d < config.min_palm_scale:
        return None
    return float(math.degrees(math.acos(min(length_2d / length_3d, 1.0))))


def is_palm_tilted(
    observation: HandObservation,
    config: GestureConfig = DEFAULT_GESTURE_CONFIG,
) -> bool:
    """자세 판정을 거부해야 할 만큼 손이 기울었는가.

    기울기를 모르면(소스가 z를 못 냄) **거부하지 않는다** — 알 수 없는 것을 위험으로
    간주해 전부 막으면 z 없는 소스에서 시스템이 통째로 멈춘다. 게이트가 없는 것과 같은
    동작이 되며, 그 사실은 `palm_tilt_degrees is None`으로 상위 계층에 드러나 있다.
    """
    if config.max_palm_tilt_degrees <= 0.0 or observation.palm_tilt_degrees is None:
        return False
    return observation.palm_tilt_degrees > config.max_palm_tilt_degrees


def select_largest_hand_index(hand_landmarks: object) -> int:
    """여러 손 중 화면에서 가장 큰 손(bounding-box 넓이 최대)의 인덱스를 고른다.

    크기 = 정규화 이미지 좌표에서 21개 랜드마크의 x·y bounding-box 넓이. 카메라에
    가까운/큰 손일수록 넓이가 커지므로, MediaPipe의 트래킹 관성이나 handedness
    score와 무관하게 매 프레임 가장 큰 손을 주 조작 손으로 선택한다 — 추적 중인
    손이 있어도 더 큰 손이 등장하면 그쪽으로 전환된다. 단, 후보가 여럿 잡히려면
    config.num_hands >= 2로 검출 슬롯을 열어 둬야 한다.

    ``hand_landmarks``는 MediaPipe HandLandmarkerResult.hand_landmarks 형태 —
    손마다 .x/.y를 가진 랜드마크 리스트의 시퀀스. 비어 있으면 0을 돌려준다
    (호출 측이 이미 빈 결과를 걸러낸 뒤 부른다).
    """
    if not hand_landmarks:
        return 0
    best_index = 0
    best_area = -1.0
    for index, landmarks in enumerate(hand_landmarks):
        if not landmarks:
            continue
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best_index = index
    return best_index


def _lost_tracking_observation(timestamp_ms: int, frame_id: int) -> HandObservation:
    """손을 찾지 못한(또는 퇴화한) 프레임의 관측값 — 추적 손실을 지어내지 않는다."""
    return HandObservation(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        landmarks=np.zeros((HAND_LANDMARK_COUNT, LANDMARK_DIMS), dtype=np.float64),
        handedness="",
        palm_scale=0.0,
        detection_confidence=0.0,
        handedness_score=0.0,
        hand_detected=False,
        wrist_position=np.zeros(LANDMARK_DIMS, dtype=np.float64),
    )


def normalize_hand(
    raw: RawHandLandmarks,
    config: GestureConfig = DEFAULT_GESTURE_CONFIG,
) -> HandObservation:
    """원시 랜드마크를 손목 기준·손바닥 크기 정규화한 HandObservation으로 변환한다.

    손바닥 크기(root→tip 거리)가 `config.min_palm_scale` 미만이면 퇴화한
    landmark로 보고 추적 손실 관측값을 반환한다 — 0에 가까운 값으로 나눠 좌표가
    폭주하는 것을 막는다(development-principles.md 7.2: 모델 출력을 그대로 믿지 않음).
    """
    if raw.detection_confidence < config.min_hand_detection_confidence:
        return _lost_tracking_observation(raw.timestamp_ms, raw.frame_id)

    scale_root = raw.points[config.palm_scale_root_index]
    scale_tip = raw.points[config.palm_scale_tip_index]
    palm_scale = float(np.linalg.norm(scale_tip - scale_root))
    if not math.isfinite(palm_scale) or palm_scale < config.min_palm_scale:
        return _lost_tracking_observation(raw.timestamp_ms, raw.frame_id)

    # 1) 손목 기준 좌표 정규화: config.origin_index(기본: 손목)를 원점으로 옮긴다.
    # 2) 손바닥 크기 정규화: 손바닥 기준 거리로 나눠 카메라 거리 의존을 제거한다.
    #    원점(origin_index)과 스케일 기준(palm_scale_root_index)은 별개 설정이다.
    origin = raw.points[config.origin_index]
    normalized = (raw.points - origin) / palm_scale
    # 손목 평행이동 신호: 원점화하지 않고 palm_scale로만 정규화해 손 전체의 위치를
    # 보존한다(카메라 거리 독립). normalized에서는 손목이 항상 (0,0,0)이라 이 신호가
    # 사라지므로, 여기서 따로 남겨 feature 계층이 미분한다.
    wrist_position = origin / palm_scale

    return HandObservation(
        timestamp_ms=raw.timestamp_ms,
        frame_id=raw.frame_id,
        landmarks=normalized.astype(np.float64, copy=False),
        handedness=raw.handedness,
        palm_scale=palm_scale,
        detection_confidence=raw.detection_confidence,
        handedness_score=raw.handedness_score,
        hand_detected=True,
        wrist_position=wrist_position.astype(np.float64, copy=False),
        palm_tilt_degrees=raw.palm_tilt_degrees,
    )


class HandLandmarkSource(Protocol):
    """프레임 → HandObservation 을 내는 교체 가능한 landmark 소스 경계.

    MVP는 `mediapipe_hands.MediaPipeHandLandmarker`가 구현한다. 나중에 keypoint를
    WebSocket으로 GPU 서버에 보내는 원격 소스로 바꾸더라도, downstream(feature
    engineering·TCN/GRU·fusion)은 이 Protocol만 바라보므로 수정할 필요가 없다
    (2026-07-18 결정: 추론 위치를 교체 가능한 경계로 분리).
    """

    def process(
        self,
        rgb_frame: npt.NDArray[np.uint8],
        timestamp_ms: int,
        frame_id: int,
    ) -> HandObservation:
        """RGB 프레임 하나를 처리해 정규화된 HandObservation을 반환한다."""
        ...
