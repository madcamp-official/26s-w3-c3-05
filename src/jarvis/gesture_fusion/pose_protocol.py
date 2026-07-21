"""정적 손 자세 분류의 교체 가능한 경계 — torch에 의존하지 않는 순수 모듈.

`model_protocol.py`(동적 제스처 TCN)와 같은 구조이자 **별개의 관심사**다:

    model_protocol  : 시퀀스 → swipe/rotate 등 동적 제스처 (서버 이관 대상)
    pose_protocol   : 단일 프레임 → 정적 손 모양 (로컬, 커서·클릭·스크롤 판정)

두 모델은 입력 차원도 산출물도 다르며 서로를 import하지 않는다. 여기 있는 타입은
torch 없이 검증·테스트할 수 있고, 실제 구현은 `pose_classifier.py`(`ml` extra)에 있다.

**기울기 신뢰 판정**이 이 계층의 핵심 책임이다. 기울기 내성은 자세마다 크게 달라
(실측 20~30° 구간: two_fingers 100%, index_point 0%) 전역 임계 하나로는 스크롤 자세를
막거나 위험한 자세를 통과시키게 된다. 그래서 **분류한 뒤** 예측된 자세별 허용 각도로
신뢰 여부를 판정한다 — 순환 논리가 아니다. 먼저 분류하고, 그 결과를 믿어도 되는지를
데이터로 만든 표에서 확인하는 순서다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

NONE_POSE = "none"
"""제어 명령이 아님. 이 판정이 나오면 어떤 동작도 실행하지 않는다."""

# 수집·학습에서 쓰는 정적 자세 라벨. 순서가 분류기 출력 인덱스와 일치해야 한다
# (학습 산출물의 `label_names`가 진실이며, 로드 시 대조한다).
DEFAULT_POSE_LABELS: tuple[str, ...] = (
    "index_point",
    "pinch_index",
    "pinch_middle",
    "two_fingers",
    "open_palm",
    "fist",
    # 제어 자세가 아닌 모든 손 상태(전이 구간, 휴지, 일상 동작). 이게 없으면 분류기가
    # 매 프레임 억지로 6개 중 하나를 골라, 손이 화면에 보이기만 하면 명령이 나간다.
    # 실측 효과(2026-07-20): 우클릭 직전의 엄지-중지 접근 구간이 two_fingers로 분류돼
    # 스크롤이 오발동하던 문제가 0건이 됐고, 헐거운 핀치(pinch_index 23.7%,
    # pinch_middle 32.3%)가 none으로 흡수됐다. 전체 정확도는 93.9%→87.1%로 내려가나
    # 오류가 안전한 쪽으로 옮겨간 결과다: 오발동 1.7%, 명령 간 오인 2.8%, 놓침 15.4%.
    NONE_POSE,
)

# 자세별 기울기 허용 각도(도) — 2026-07-20 실측 기반.
#
#   자세          0~20°   20~30°   30~90°
#   two_fingers   100%     100%      99%   → 손가락을 편 자세는 기울어도 실루엣이 남는다
#   open_palm      99%     100%       -
#   pinch_index   100%       -        -
#   pinch_middle   86%       -        0%
#   fist           77%       -        -
#   index_point    62%       0%       -    → 기울면 다른 자세와 뭉개진다
#
# 근거가 있는 구간까지만 열어준다. 표본이 없어 확인 못 한 구간은 보수적으로 20°에
# 둔다 — 모르는 각도를 임의로 허용하면 조용히 오동작한다.
DEFAULT_POSE_TILT_LIMITS: dict[str, float] = {
    "index_point": 20.0,
    # 중지 하나 → 탭 닫기(파괴적). 명시 엔트리가 없으면 unknown-label 폴백으로 가장
    # 보수적인 20°가 걸려, 분류기가 맞힌 진짜 middle_point의 56%까지 조용히 거부됐다
    # (2026-07-22 v2 데이터 실측: 20°면 진짜 44%만 통과). 진짜 중앙 기울기 21° vs
    # none→middle_point 오분류 중앙 37°이라, 30°가 진짜 84% 유지·오분류 71% 차단으로
    # 최적. dwell 250ms와 곱해져 실오발동은 더 낮다.
    "middle_point": 30.0,
    "pinch_index": 20.0,
    "pinch_middle": 20.0,
    # 편 두 손가락은 기울어도 실루엣이 남아 30~90° 묶음에서도 99% 정확하다(위 표). 40°
    # 초과 구간은 표본이 적었지만 오분류가 관찰되지 않아, 스크롤 자세의 기울기 제한을
    # 없앤다(90° = 사실상 무제한). 스크롤은 아래를 크게 가리키는 자세라 tilt 게이트가
    # 오히려 정상 조작을 막던 문제를 해소한다.
    "two_fingers": 90.0,
    "open_palm": 30.0,
    "fist": 20.0,
    # none은 "명령 아님"이라 기울기와 무관하게 유효하다 — 기울었다고 거부하면
    # 그 프레임이 다시 명령 후보가 되어버린다(거부의 방향이 반대다).
    NONE_POSE: 90.0,
}


# 손끝 랜드마크(엄지·검지·중지·약지·새끼). 이들 사이의 쌍거리를 feature에 더한다.
FINGERTIPS: tuple[int, ...] = (4, 8, 12, 16, 20)


def pose_features(landmarks: FloatArray) -> FloatArray:
    """정규화된 (21, D) 좌표 → 분류기 입력 벡터. **학습과 추론이 같이 쓴다.**

    좌표만 넣으면 손끝 사이의 *관계*를 모델이 스스로 뽑아내야 하는데, 작은 MLP는 그걸
    잘 못 한다. 엄지-중지끝 거리는 단독으로도 index_point와 pinch_middle을 오류율
    10.9%로 가르지만(중앙 0.254 vs 0.075), 좌표만 준 모델의 index_point 재현율은
    50.2%였다. 손끝 쌍거리 10개를 명시적으로 더하자 전체 82.6%→92.3%,
    index_point 50.2%→94.0%, fist 80.5%→99.9%로 올랐다(2026-07-20 실측).

    이 함수가 학습·추론의 단일 진실이다 — 한쪽만 바뀌면 예외 없이 정확도만 떨어지는,
    가장 찾기 어려운 고장이 된다. 저장 파일의 `input_dim`이 그 어긋남을 잡아준다.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"landmarks must be 2-D (21, D), got shape {points.shape}")
    distances = [
        float(np.linalg.norm(points[FINGERTIPS[a]] - points[FINGERTIPS[b]]))
        for a in range(len(FINGERTIPS))
        for b in range(a + 1, len(FINGERTIPS))
    ]
    return np.concatenate([points.reshape(-1), np.asarray(distances, dtype=np.float64)])


def pose_feature_dimension(landmark_count: int, landmark_dims: int) -> int:
    """`pose_features`가 내는 차원 — 모델 shape 검증용."""
    pairs = len(FINGERTIPS) * (len(FINGERTIPS) - 1) // 2
    return landmark_count * landmark_dims + pairs


@dataclass(frozen=True, slots=True)
class PosePrediction:
    """한 프레임의 자세 판정 결과.

    `trusted=False`는 "자세를 모른다"가 아니라 **"이 판정을 실행에 쓰지 말라"**는
    뜻이다. `label`·`confidence`는 그대로 담겨 있어 디버깅 툴이 무엇이 왜 거부됐는지
    보여줄 수 있다 — 거부를 조용히 감추면 사용자는 손을 세울 기회를 얻지 못한다.
    """

    label: str
    confidence: float
    trusted: bool
    reason: str = ""
    palm_tilt_degrees: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")
        if self.trusted and self.reason:
            raise ValueError("trusted prediction must not carry a rejection reason")
        if not self.trusted and not self.reason:
            raise ValueError("rejected prediction must state a reason")


def is_pose_trusted(
    label: str,
    palm_tilt_degrees: float | None,
    tilt_limits: dict[str, float] | None = None,
) -> tuple[bool, str]:
    """예측된 자세를 이 기울기에서 믿어도 되는가. (신뢰여부, 거부사유)를 돌려준다.

    기울기를 모르면(소스가 z를 못 냄) 막지 않는다 — 알 수 없음을 위험으로 간주해
    전부 거부하면 z 없는 소스에서 시스템이 통째로 멈춘다. 모르는 자세 라벨은 가장
    보수적인 한계를 적용한다(임의 허용보다 거부가 안전하다).
    """
    limits = DEFAULT_POSE_TILT_LIMITS if tilt_limits is None else tilt_limits
    if palm_tilt_degrees is None:
        return True, ""
    limit = limits.get(label, min(limits.values()) if limits else 0.0)
    if palm_tilt_degrees <= limit:
        return True, ""
    return False, f"기울기 {palm_tilt_degrees:.0f}° > {label} 허용 {limit:.0f}°"


class PoseClassifier(Protocol):
    """정규화된 랜드마크 한 프레임 → 자세 판정.

    구현체를 원격 추론이나 다른 아키텍처로 바꿔도 호출 측은 이 Protocol만 본다.
    """

    def classify(
        self, landmarks: FloatArray, palm_tilt_degrees: float | None
    ) -> PosePrediction:
        """`landmarks`는 정규화·평활된 (21, LANDMARK_DIMS) 좌표(학습과 동일 전처리)."""
        ...


@dataclass(frozen=True, slots=True)
class NullPoseClassifier:
    """모델이 없을 때 쓰는 구현 — 항상 거부한다.

    자세를 지어내지 않는다. 모델 파일이 없다는 사실이 UI에 그대로 드러나야, 학습을
    안 돌린 상태를 "인식이 잘 안 된다"로 오해하지 않는다.
    """

    reason: str = "자세 분류 모델 없음"

    def classify(
        self, landmarks: FloatArray, palm_tilt_degrees: float | None
    ) -> PosePrediction:
        del landmarks
        return PosePrediction(
            label="",
            confidence=0.0,
            trusted=False,
            reason=self.reason,
            palm_tilt_degrees=palm_tilt_degrees,
        )
