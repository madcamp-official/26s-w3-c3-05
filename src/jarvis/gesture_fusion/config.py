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

    # --- Landmark 평활화 (README 8장 "속도·가속도" 앞단, 1€ 필터) ---
    # 위치 좌표의 프레임별 고주파 지터를 미분(속도·가속도) 전에 줄인다. 지터를 그대로
    # 두면 이산 미분이 노이즈를 증폭해 모델 입력이 크게 흔들리므로, 여기서 One-Euro
    # 필터로 평활화한다. 값을 바꾸면 documents/gesture-fusion.md·decisions.md에 기록하고,
    # **학습 데이터도 같은 설정으로 전처리해야** 추론과 일관된다(모델 재현성).
    smooth_landmarks: bool = True
    """정규화된 랜드마크를 속도·가속도 계산 전에 One-Euro로 평활화할지 여부."""

    smoothing_min_cutoff: float = 1.0
    """기본 평활 강도(Hz). 낮을수록 정지 시 더 부드럽지만 지연이 커진다."""

    smoothing_beta: float = 0.5
    """속도에 따른 컷오프 개방 계수. 높을수록 빠른 동작에서 지연이 줄어든다."""

    smoothing_d_cutoff: float = 1.5
    """내부 속도 추정의 평활 컷오프(Hz)."""

    # --- Feature engineering (README 8장 "속도·관절 각도 생성") ---
    # 어떤 feature 그룹을 모델 입력 벡터에 넣을지 켜고 끈다. 모델을 갈아끼우거나
    # 입력 차원을 줄일 때 코드 수정 없이 조절한다. 순서(위치→각도→속도)는
    # 고정이며, 켜진 그룹만 순서대로 이어붙인다.
    #
    # 손가락 관절 위치의 가속도(2026-07-19 이전 include_acceleration 필드)는 모델
    # 입력에서 완전히 제거했다(2026-07-19 결정, documents/decisions.md). 손목
    # 평행이동 가속도(아래 include_wrist_translation의 wrist_acceleration)는 별개
    # 신호(swipe 판별에 필요)라 유지한다 — 혼동하지 말 것.
    include_positions: bool = True
    """정규화된 21개 랜드마크 좌표(63차원)를 feature에 포함한다."""

    include_joint_angles: bool = True
    """손가락 관절 굴곡각(JOINT_ANGLE_TRIPLETS 기준)을 feature에 포함한다."""

    include_velocity: bool = True
    """프레임 간 좌표 속도(초당, causal 차분)를 feature에 포함한다."""

    include_wrist_translation: bool = True
    """손목의 정규화된 평행이동 속도·가속도(각 3차원, 합 6차원)를 feature에 포함한다.

    손목 기준 정규화(`origin_index`)는 매 프레임 손목을 원점(0,0,0)으로 옮기므로,
    손 모양·회전이 그대로인 순수 평행이동(swipe)에서는 position·velocity·acceleration이
    21개 랜드마크 전부 이론상 0이 된다 — swipe_up/down/left/right를 구분할 유일한 신호
    (손 전체의 이동 방향·속력)가 feature에서 사라진다. 이 그룹은 원점화하지 않고
    palm_scale로만 정규화한 손목 좌표(`HandObservation.wrist_position`, 카메라 거리에
    독립)를 causal 차분해 그 이동 속도·가속도를 되살린다. 위치 자체가 아니라 속도·가속도만
    넣어 "프레임 내 어디에 있느냐"가 아니라 "어느 방향으로 얼마나 움직이느냐"만 담는다
    (프레임 위치 overfitting 방지). rotate 계열은 회전이 보존돼 영향받지 않는다
    (documents/decisions.md 2026-07-19). 학습 데이터도 반드시 같은 설정으로 전처리해야 한다."""

    max_frame_gap_ms: int = 200
    """이 간격을 넘는 프레임 공백이면 속도·가속도 history를 리셋한다.

    추적이 잠깐 끊겼다 돌아온 두 프레임 사이의 큰 점프를 실제 손 움직임으로
    오해해 허위 속도를 만드는 것을 막는다(development-principles.md 2·5절: 불확실하면
    지어내지 않는다). 리셋 후 첫 프레임의 속도·가속도는 0이다.
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
        if self.max_frame_gap_ms <= 0:
            raise ValueError("max_frame_gap_ms must be positive")
        if not math.isfinite(self.smoothing_min_cutoff) or self.smoothing_min_cutoff <= 0.0:
            raise ValueError("smoothing_min_cutoff must be finite and positive")
        if not math.isfinite(self.smoothing_d_cutoff) or self.smoothing_d_cutoff <= 0.0:
            raise ValueError("smoothing_d_cutoff must be finite and positive")
        if not math.isfinite(self.smoothing_beta) or self.smoothing_beta < 0.0:
            raise ValueError("smoothing_beta must be finite and non-negative")
        if not any(
            (
                self.include_positions,
                self.include_joint_angles,
                self.include_velocity,
                self.include_wrist_translation,
            )
        ):
            raise ValueError("at least one feature group must be enabled")


# MediaPipe Hand Landmarker는 손 하나당 21개 랜드마크를 낸다. downstream feature
# engineering·gesture 로직이 참조하도록 표준 인덱스를 한 곳에 둔다.
# 참고: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
HAND_LANDMARK_COUNT = 21

# 각 랜드마크의 좌표 차원. MediaPipe는 (x, y, z)를 주지만 z(깊이)는 단안 웹캠에서
# 추정한 값이라 노이즈가 크고 프레임마다 크게 흔들린다(특히 손가락이 카메라 쪽으로
# 접히는 주먹류 동작). 검출 안정성을 위해 x·y 2D만 사용한다. 이 값 하나가 원시 좌표
# 추출(mediapipe_hands·hand_probe)부터 정규화(landmarks)·feature 차원(features)까지
# 전 파이프라인의 좌표 차원을 결정한다 — 3으로 되돌리면 다시 z를 포함하지만, 학습
# 데이터도 같은 차원으로 전처리해야 하므로(모델 재현성) 변경 시 반드시 재학습한다.
LANDMARK_DIMS = 2

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


# 손가락 관절 굴곡각을 재는 (a, b, c) 랜드마크 삼각. b가 꼭짓점이고, 벡터 b→a와
# b→c 사이 각을 잰다. 각 손가락마다 뿌리(손목/MCP)부터 끝까지 굽힘 정도를 담아
# "몇 손가락을 폈는지"·pinch·주먹 같은 정적 손 모양(README 8장 지원 제스처)을
# 표현한다. 좌표계·모델을 바꿔도 이 정의만 손대면 각도 feature가 따라온다.
JOINT_ANGLE_TRIPLETS: tuple[tuple[int, int, int], ...] = (
    # Thumb
    (WRIST, THUMB_MCP, THUMB_IP),
    (THUMB_MCP, THUMB_IP, THUMB_TIP),
    # Index
    (WRIST, INDEX_FINGER_MCP, INDEX_FINGER_PIP),
    (INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_DIP),
    (INDEX_FINGER_PIP, INDEX_FINGER_DIP, INDEX_FINGER_TIP),
    # Middle
    (WRIST, MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP),
    (MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP),
    (MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP, MIDDLE_FINGER_TIP),
    # Ring
    (WRIST, RING_FINGER_MCP, RING_FINGER_PIP),
    (RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_DIP),
    (RING_FINGER_PIP, RING_FINGER_DIP, RING_FINGER_TIP),
    # Pinky
    (WRIST, PINKY_MCP, PINKY_PIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP),
    (PINKY_PIP, PINKY_DIP, PINKY_TIP),
)


DEFAULT_GESTURE_CONFIG = GestureConfig()
