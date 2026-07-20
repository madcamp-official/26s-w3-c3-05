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

# 수집·학습에서 쓰는 정적 자세 라벨. 순서가 분류기 출력 인덱스와 일치해야 한다
# (학습 산출물의 `label_names`가 진실이며, 로드 시 대조한다).
DEFAULT_POSE_LABELS: tuple[str, ...] = (
    "index_point",
    "pinch_index",
    "pinch_middle",
    "two_fingers",
    "open_palm",
    "fist",
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
    "pinch_index": 20.0,
    "pinch_middle": 20.0,
    "two_fingers": 30.0,
    "open_palm": 30.0,
    "fist": 20.0,
}


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
