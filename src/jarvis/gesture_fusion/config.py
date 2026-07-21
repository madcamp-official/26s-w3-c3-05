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
    num_hands: int = 2
    """동시에 검출할 최대 손 개수(검출 슬롯 상한).

    주 조작 손은 이 중 bounding-box가 가장 큰(카메라에 가까운) 손 하나를
    landmarks.select_largest_hand_index로 고른다. 새로 등장한 더 큰 손으로
    전환되려면 슬롯이 최소 2개는 열려 있어야 한다 — 1이면 MediaPipe가 기존 손을
    트래킹 관성으로 붙잡아 새 손이 검출조차 안 된다. 화면에 손이 더 많을 수 있는
    환경이면 이 값을 올린다(손마다 landmark 추론이 돌아 프레임당 연산량은 는다)."""

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

    smoothing_min_cutoff: float = 2.0
    """기본 평활 강도(Hz). 낮을수록 정지 시 더 부드럽지만 지연이 커진다.

    2026-07-20: 손 추적이 "느리게 따라온다"는 사용자 보고로 1.0→2.0 상향(위상 지연
    축소, 정지 시 잔떨림은 약간 증가하는 트레이드오프). 실사용 확인 대기 중."""

    smoothing_beta: float = 1.2
    """속도에 따른 컷오프 개방 계수. 높을수록 빠른 동작에서 지연이 줄어든다.

    2026-07-20: 위 min_cutoff와 같은 이유로 0.5→1.2 상향. 실사용 확인 대기 중."""

    smoothing_d_cutoff: float = 1.5
    """내부 속도 추정의 평활 컷오프(Hz)."""

    # --- palm_scale 평활화 (2026-07-19, 손목 평행이동 잡음 수정) ---
    # `wrist_position = origin / palm_scale`은 분자(화면상 절대 위치, 대략 0.3~0.7)가
    # 일반 landmark의 분자(손 안에서의 상대적 차이, 손목 자신은 0)보다 훨씬 커서,
    # 매 프레임 새로 계산되는 palm_scale의 잡음이 나눗셈을 타고 훨씬 크게 증폭된다.
    # 정지한 손 시뮬레이션 실측: 이 증폭 때문에 손목 속도 잡음이 손가락 끝 속도 잡음보다
    # 약 2.5배 컸다(위 smoothing_* 만으로는 못 잡음 — 그 필터는 나눗셈 *이후* 값에만
    # 적용됨). palm_scale 자체를 별도로 평활화해 나눗셈에 쓰면(기존 필터는 그대로 두고
    # 추가) 정지 시 잡음이 약 3.85배 줄어든다(문서화된 실측: 0.81 → 0.21 palm-width/s).
    # `normalize_hand`는 순수 함수라 여기서 palm_scale을 평활화할 수 없으므로,
    # `HandFeatureExtractor`가 raw palm_scale과 평활화된 palm_scale의 비율로
    # `wrist_position`을 재조정한다(`origin`에 직접 접근하지 않고도 재현 가능:
    # `wrist_position × raw_palm_scale / smoothed_palm_scale = origin / smoothed_palm_scale`).
    smooth_palm_scale: bool = True
    """`smooth_landmarks`와 함께 켜진다(같은 "미분 전 평활화" 전략의 일부).
    별도 토글이 필요하면 이 필드를 독립시킨다."""

    palm_scale_smoothing_min_cutoff: float = 1.0
    """palm_scale 평활 강도(Hz). 손 크기는 카메라 거리 변화 외엔 천천히 바뀌므로
    낮은 값으로도 충분히 안정적이다."""

    palm_scale_smoothing_beta: float = 0.0
    """palm_scale 변화 속도에 따른 컷오프 개방 계수. 0 = 속도 적응 없이 고정 컷오프로만
    평활화(단순 저역통과) — palm_scale은 랜드마크 위치와 달리 "빠른 동작"이라는
    개념이 없어 속도 적응이 불필요하다는 실측 결과를 반영."""

    palm_scale_smoothing_d_cutoff: float = 1.0
    """palm_scale 내부 변화율 추정의 평활 컷오프(Hz)."""

    # --- 손 기울기 게이트 ---
    max_palm_tilt_degrees: float = 20.0
    """이 각도를 넘게 기울어진 손은 자세 판정을 **거부**한다(0이면 게이트 없음).

    각도는 손바닥 축(손목→중지 MCP)이 이미지 평면과 이루는 각이다. 손을 카메라 쪽으로
    눕히면 이 축이 2D에서 단축되어 자세 정보가 실제로 소실된다 — 좌표를 어떻게 정규화해도
    복원되지 않는다(2026-07-20 측정: 정규화 방식 4종이 정확도에서 통계적 동률).

    측정된 구간별 분류 정확도(6클래스, 에피소드 단위 홀드아웃):
         0~10°  90.8%     20~30°  47.3%
        10~20°  76.6%     30~90°  37.0%
    기울기 구간만 따로 학습해도 10~20°에서 39.8%, 20° 초과에서 25.9%(우연 17%)라
    데이터를 더 모아 해결되는 문제가 아니다.

    임계 20°는 판정률 85.1%에 정확도 88.7%인 지점이다(15°는 80.6%/90.0%, 30°는
    92.0%/85.5%). 실제 조작에서 손을 화면과 완벽히 나란히 유지할 수 없으므로 어느 정도
    기울기는 허용해야 한다는 요구와, 20° 초과에서 급락하는 성능 사이의 타협점이다.

    거부는 조용히 무시하지 말고 사용자에게 표시해야 한다 — 왜 반응이 없는지 알아야
    손을 세울 수 있다(development-principles.md: 실패를 감추지 않는다).
    """

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

    include_palm_orientation: bool = False
    """손바닥 삼각형의 **부호 있는** 면적과 그 변화율(2차원)을 feature에 포함한다.

    회전(`rotate_clockwise`/`rotate_counter_clockwise`) 판별용으로 설계했으나, A/B
    학습 결과 효과가 없어 기본값을 off로 둔다(2026-07-20 추가·검증).

    **A/B 결과(기본 파이프라인 대비, 재학습·같은 데이터)**: 개선하려던 회전 두
    클래스의 F1이 오히려 -0.0025 / -0.0003, 상호 혼동은 3,356 -> 3,389건으로 증가,
    배경합산 macro-F1도 0.8052 -> 0.8042. 전부 노이즈 범위이거나 소폭 악화다.

    **왜 남겨두는가**: 오프라인 분리도(clip 단위 로지스틱 회귀 82.8% vs 각속도 75%)는
    분명했는데도 frame-level 모델 학습으로 전달되지 않았다 —
    `velocity_smoothing_window`와 같은 패턴의 두 번째 사례다. "정적 신호 분리도"와
    "causal frame-level 학습 가능성"이 별개임을 보여주는 기록으로 코드를 남긴다.
    회전 문제의 근본 원인은 팔뚝축 회전이 화면 밖에서 일어나 `LANDMARK_DIMS = 2`가
    z를 버린 것이라, 진짜 해결은 z 복원(재추출 필요)일 가능성이 높다.

    **문제**: 두 회전 클래스가 서로에게 각각 21.9%/20.4%를 잃어(총 3,334건, 최대
    오분류원) F1이 0.71~0.73에 머문다. 원인은 모델 용량이 아니라 **전처리에서
    정보가 이미 삭제된 것**이다 — Jester "Turning Hand"는 팔뚝축(카메라 방향) 회전이
    대부분이라, 화면상 순회전은 34~39°뿐이고 손바닥 폭 변동계수만 0.41~0.43으로
    다른 클래스의 2배다(강체 in-plane 회전이면 폭이 불변이어야 한다 = foreshortening).
    `LANDMARK_DIMS = 2`로 z를 버리므로 회전축 성분이 남지 않는다.

    **해법**: 손목·검지MCP·소지MCP가 이루는 삼각형의 부호 있는 면적은 손바닥/손등 중
    어느 쪽이 카메라를 향하는지에 따라 부호가 뒤집힌다. 팔뚝축 회전은 시계/반시계에
    따라 그 뒤집힘이 반대 순서로 일어나므로, **면적 변화율의 부호가 곧 회전 방향**이다
    — z 없이 2D 투영만으로 축 회전 방향을 복원하는 관측량이다.

    실측(클립 단위, held-out 720개 로지스틱 회귀): 부호면적 변화율 단독 82.8% 대
    각속도 75.1%·누적회전 75.6%. 넷을 다 합쳐도 82.4%로 더 오르지 않아, 이 신호가
    기존 회전 계열 정보를 사실상 포함한다.

    좌표(42dim)로부터 원리적으로는 유도 가능하지만 부호면적은 좌표들의 **이차 항**
    (외적)이라, 채널 32개·29K 파라미터 ReLU 네트워크가 곱셈 상호작용을 근사하는 것은
    비효율적이다 — 명시적으로 넣는다. 캐시된 랜드마크에서 계산되므로 재추출은
    필요 없다(재학습은 필요)."""

    max_frame_gap_ms: int = 200
    """이 간격을 넘는 프레임 공백이면 속도·가속도 history를 리셋한다.

    추적이 잠깐 끊겼다 돌아온 두 프레임 사이의 큰 점프를 실제 손 움직임으로
    오해해 허위 속도를 만드는 것을 막는다(development-principles.md 2·5절: 불확실하면
    지어내지 않는다). 리셋 후 첫 프레임의 속도·가속도는 0이다.
    """

    velocity_smoothing_window: int = 1
    """`include_velocity` 속도(초당 좌표 차분)를 인접 몇 프레임 평균으로 낼지(causal,
    미래 프레임 안 봄). 1=평균 없음(기존과 동일한 두 프레임 차분).

    2026-07-20 실험(오프라인, `training/cache` 랜드마크로 측정): 두 손가락 슬라이드의
    위/아래 방향 분리도(Cohen's d)가 좌/우(1.83)보다 훨씬 낮았다(0.50) — down 방향
    속도의 분산이 up의 3배로 유독 크다. Window=9 causal 평균을 적용하면 위/아래
    분리도가 0.50→1.25로(+150%), 좌/우도 1.83→2.07로 함께 개선됐다. 원인은 아래
    슬라이드가 중력 보조로 더 빠르고 덜 제어돼(모션 블러·검출 지터 증가) 프레임 간
    단순 차분의 잡음이 큰 것으로 추정 — augmentation·fps 문제가 아니라 신호 자체의
    잡음이라 스무딩이 직접적인 대응이다. 이 값을 바꾸면 학습 데이터 재추출은
    필요 없지만(캐시는 raw landmark) 재학습은 필요하다."""

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
        if not math.isfinite(self.max_palm_tilt_degrees) or not 0.0 <= self.max_palm_tilt_degrees <= 90.0:
            raise ValueError("max_palm_tilt_degrees must be finite and within [0, 90]")
        if self.velocity_smoothing_window < 1:
            raise ValueError("velocity_smoothing_window must be at least 1")
        if not math.isfinite(self.smoothing_min_cutoff) or self.smoothing_min_cutoff <= 0.0:
            raise ValueError("smoothing_min_cutoff must be finite and positive")
        if not math.isfinite(self.smoothing_d_cutoff) or self.smoothing_d_cutoff <= 0.0:
            raise ValueError("smoothing_d_cutoff must be finite and positive")
        if not math.isfinite(self.smoothing_beta) or self.smoothing_beta < 0.0:
            raise ValueError("smoothing_beta must be finite and non-negative")
        if not math.isfinite(self.palm_scale_smoothing_min_cutoff) or self.palm_scale_smoothing_min_cutoff <= 0.0:
            raise ValueError("palm_scale_smoothing_min_cutoff must be finite and positive")
        if not math.isfinite(self.palm_scale_smoothing_d_cutoff) or self.palm_scale_smoothing_d_cutoff <= 0.0:
            raise ValueError("palm_scale_smoothing_d_cutoff must be finite and positive")
        if not math.isfinite(self.palm_scale_smoothing_beta) or self.palm_scale_smoothing_beta < 0.0:
            raise ValueError("palm_scale_smoothing_beta must be finite and non-negative")
        if not any(
            (
                self.include_positions,
                self.include_joint_angles,
                self.include_velocity,
                self.include_wrist_translation,
                self.include_palm_orientation,
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
