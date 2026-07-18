"""Named, tunable parameters for the Gesture & Intent Fusion module.

README 8장 "Dynamic Gesture Spotter"·9장 "Multimodal Intent Fusion"에 쓰이는
임계값과 전처리 파라미터를 코드의 단일 기준으로 모은다. 값을 바꿀 때는
documents/gesture-fusion.md와 documents/decisions.md도 함께 갱신한다
(development-principles.md 1.2·8절: 데모 시나리오만 통과하는 하드코딩 금지).

이 파일은 "파라미터/모델을 코드 수정 없이 갈아끼운다"는 설계 목표의 핵심이다.
landmark 소스(MediaPipe든 원격 GPU 서버든)와 추론 모델은 이 config가 주입하는
값만 바라보고, 소스 구현 자체는 `HandLandmarkSource` Protocol로 교체한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GestureConfig:
    """손 랜드마크 전처리와 제스처 spotting에 쓰는 튜너블 파라미터.

    Task 1(hand landmark)에서 실제로 사용하는 필드만 먼저 둔다. feature
    engineering·TCN/GRU·fusion이 붙으면서 필드가 늘어나며, 각 추가는
    documents/gesture-fusion.md에 의미·단위와 함께 기록한다.
    """

    # --- Landmark 소스 (README 8장 "MediaPipe Hand Landmark") ---
    num_hands: int = 1
    """동시에 추적할 최대 손 개수. MVP는 1(주 조작 손)."""

    min_hand_detection_confidence: float = 0.5
    """이 값 미만의 검출 신뢰도는 손 없음(추적 손실)으로 취급한다."""

    min_hand_presence_confidence: float = 0.5
    """프레임 내 손 존재 신뢰도 하한 (MediaPipe Hand Landmarker 옵션)."""

    min_tracking_confidence: float = 0.5
    """프레임 간 추적 신뢰도 하한 (MediaPipe Hand Landmarker 옵션)."""

    # --- 좌표계 원점 (README 8장 "손목 기준 좌표 정규화") ---
    origin_index: int = 0
    """평행이동으로 원점(0,0,0)에 놓을 랜드마크 (기본: WRIST).

    스케일 기준(`palm_scale_root_index`)과 **의도적으로 분리된** 필드다. 스케일을
    다른 두 점으로 재도록 튜닝하더라도 좌표 원점은 여기 값에만 따르므로, 한 필드를
    바꿨을 때 원점이 딸려 이동하는 숨은 커플링이 생기지 않는다.
    """

    # --- 손바닥 크기 정규화 (README 8장 "손바닥 크기 정규화") ---
    palm_scale_root_index: int = 0
    """손바닥 크기 기준 벡터의 시작점 랜드마크 (기본: WRIST).

    스케일 계산에만 쓰인다 — 좌표 원점은 `origin_index`가 따로 정한다.
    """

    palm_scale_tip_index: int = 9
    """손바닥 크기 기준 벡터의 끝점 랜드마크 (기본: MIDDLE_FINGER_MCP).

    `palm_scale_root_index`와의 거리로 손 크기를 정규화한다. 손목→중지 MCP는
    손가락 접힘·펼침과 무관해 크기 기준으로 안정적이다. 카메라 거리에 따른
    스케일 변화를 제거하는 것이 목적이며, 회전 정규화는 하지 않는다(손목 회전
    제스처 신호를 지우지 않기 위해, README 8장 "볼륨 조절: 손목 회전").
    """

    min_palm_scale: float = 1e-4
    """이 값 미만의 손바닥 크기는 퇴화(landmark 붕괴)로 보고 추적 손실 처리한다.

    0에 가까운 값으로 나눠 좌표가 폭주하는 것을 막는 안전 하한이다.
    """

    def __post_init__(self) -> None:
        if self.num_hands < 1:
            raise ValueError("num_hands must be at least 1")
        confidence_fields = {
            "min_hand_detection_confidence": self.min_hand_detection_confidence,
            "min_hand_presence_confidence": self.min_hand_presence_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
        }
        for name, value in confidence_fields.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1], got {value}")
        index_fields = {
            "origin_index": self.origin_index,
            "palm_scale_root_index": self.palm_scale_root_index,
            "palm_scale_tip_index": self.palm_scale_tip_index,
        }
        for name, value in index_fields.items():
            if not 0 <= value < HAND_LANDMARK_COUNT:
                raise ValueError(
                    f"{name} must be a valid hand landmark index [0, {HAND_LANDMARK_COUNT}), got {value}"
                )
        if self.palm_scale_root_index == self.palm_scale_tip_index:
            raise ValueError("palm_scale_root_index and palm_scale_tip_index must differ")
        if not math.isfinite(self.min_palm_scale) or self.min_palm_scale <= 0.0:
            raise ValueError("min_palm_scale must be finite and positive")


# MediaPipe Hand Landmarker는 손 하나당 21개 랜드마크를 낸다. downstream feature
# engineering·gesture 로직이 참조하도록 표준 인덱스를 한 곳에 둔다.
# 참고: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
HAND_LANDMARK_COUNT = 21

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


DEFAULT_GESTURE_CONFIG = GestureConfig()
