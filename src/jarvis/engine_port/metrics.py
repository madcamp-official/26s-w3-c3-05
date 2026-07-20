"""두 랜드마크 엔진의 출력 차이를 정량화하는 누적기.

눈으로 보는 A/B는 "느낌"밖에 안 남는다. 같은 프레임을 두 엔진에 넣었을 때 21점이
실제로 얼마나 벌어지는지, 검출 자체가 얼마나 자주 엇갈리는지를 숫자로 남긴다.

거리는 **이미지 정규화 좌표([0, 1]) 기준**이라 해상도와 무관하다. 화면 표시용
픽셀 값이 필요하면 프레임 너비를 곱하면 된다.

주의: 이 수치는 "누가 정답이냐"를 말해주지 않는다. 정답 라벨이 없으므로 두 엔진의
**불일치 정도**와 **시간적 안정성(지터)** 만 말한다. 지터는 정답 없이도 품질 신호로
쓸 수 있다 — 정지한 손에서 흔들림이 적은 쪽이 대체로 낫다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jarvis.gesture_fusion.config import HAND_LANDMARK_COUNT as LANDMARK_COUNT

FloatArray = npt.NDArray[np.float64]


def landmark_deviation(a: npt.ArrayLike, b: npt.ArrayLike) -> FloatArray:
    """두 (21, 2) 랜드마크 집합의 **점별 유클리드 거리** (21,)를 반환한다."""
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != (LANDMARK_COUNT, 2) or right.shape != (LANDMARK_COUNT, 2):
        raise ValueError(f"랜드마크는 ({LANDMARK_COUNT}, 2)여야 합니다: {left.shape}, {right.shape}")
    return np.linalg.norm(left - right, axis=1)


@dataclass
class JitterTracker:
    """연속 프레임 간 랜드마크 이동량 — 손이 멈춰 있을 때는 곧 지터다."""

    _previous: FloatArray | None = None
    _samples: list[float] = field(default_factory=list)

    def update(self, points: FloatArray | None) -> float | None:
        """이번 프레임의 평균 이동량을 기록하고 반환한다(첫 프레임/미검출이면 None)."""
        if points is None:
            self._previous = None  # 추적이 끊겼으므로 다음 프레임과 이어 붙이지 않는다
            return None
        if self._previous is None:
            self._previous = points
            return None
        movement = float(np.mean(landmark_deviation(self._previous, points)))
        self._previous = points
        self._samples.append(movement)
        return movement

    @property
    def mean(self) -> float | None:
        return float(np.mean(self._samples)) if self._samples else None


@dataclass
class ComparisonSummary:
    """비교 세션 전체의 집계 결과."""

    frames: int
    both_detected: int
    only_a: int
    only_b: int
    neither: int
    mean_deviation: float | None
    p95_deviation: float | None
    max_deviation: float | None
    per_landmark_mean: FloatArray | None
    jitter_a: float | None
    jitter_b: float | None

    @property
    def agreement_rate(self) -> float:
        """두 엔진의 검출 여부가 일치한 프레임 비율."""
        if self.frames == 0:
            return 0.0
        return (self.both_detected + self.neither) / self.frames

    def format_report(self, *, label_a: str, label_b: str, frame_width: int | None = None) -> str:
        """사람이 읽을 요약. `frame_width`를 주면 픽셀 환산도 함께 보여준다."""

        def as_text(value: float | None) -> str:
            if value is None:
                return "n/a"
            if frame_width is None:
                return f"{value:.4f}"
            return f"{value:.4f} ({value * frame_width:.1f}px)"

        lines = [
            f"총 {self.frames} 프레임",
            f"  둘 다 검출      : {self.both_detected}",
            f"  {label_a}만 검출 : {self.only_a}",
            f"  {label_b}만 검출 : {self.only_b}",
            f"  둘 다 미검출    : {self.neither}",
            f"  검출 일치율     : {self.agreement_rate:.1%}",
            "",
            "랜드마크 편차 (둘 다 검출한 프레임, 정규화 좌표)",
            f"  평균 : {as_text(self.mean_deviation)}",
            f"  p95  : {as_text(self.p95_deviation)}",
            f"  최대 : {as_text(self.max_deviation)}",
            "",
            "프레임 간 이동량 평균 (지터 지표 — 손을 멈춘 구간에서 낮을수록 안정적)",
            f"  {label_a} : {as_text(self.jitter_a)}",
            f"  {label_b} : {as_text(self.jitter_b)}",
        ]
        if self.per_landmark_mean is not None:
            worst = int(np.argmax(self.per_landmark_mean))
            lines += [
                "",
                f"편차가 가장 큰 점 : #{worst} ({as_text(float(self.per_landmark_mean[worst]))})",
            ]
        return "\n".join(lines)


class ComparisonAccumulator:
    """프레임마다 두 엔진 결과를 넣으면 집계를 유지한다."""

    def __init__(self) -> None:
        self._frames = 0
        self._both = 0
        self._only_a = 0
        self._only_b = 0
        self._neither = 0
        self._deviations: list[float] = []
        self._per_landmark_sum = np.zeros(LANDMARK_COUNT, dtype=np.float64)
        self._jitter_a = JitterTracker()
        self._jitter_b = JitterTracker()

    def update(self, points_a: FloatArray | None, points_b: FloatArray | None) -> float | None:
        """한 프레임을 반영하고, 둘 다 검출됐다면 그 프레임의 평균 편차를 반환한다."""
        self._frames += 1
        self._jitter_a.update(points_a)
        self._jitter_b.update(points_b)

        if points_a is None and points_b is None:
            self._neither += 1
            return None
        if points_b is None:
            self._only_a += 1
            return None
        if points_a is None:
            self._only_b += 1
            return None

        self._both += 1
        per_point = landmark_deviation(points_a, points_b)
        self._per_landmark_sum += per_point
        frame_mean = float(np.mean(per_point))
        self._deviations.append(frame_mean)
        return frame_mean

    def summary(self) -> ComparisonSummary:
        has_pairs = bool(self._deviations)
        return ComparisonSummary(
            frames=self._frames,
            both_detected=self._both,
            only_a=self._only_a,
            only_b=self._only_b,
            neither=self._neither,
            mean_deviation=float(np.mean(self._deviations)) if has_pairs else None,
            p95_deviation=float(np.percentile(self._deviations, 95)) if has_pairs else None,
            max_deviation=float(np.max(self._deviations)) if has_pairs else None,
            per_landmark_mean=(self._per_landmark_sum / self._both) if self._both else None,
            jitter_a=self._jitter_a.mean,
            jitter_b=self._jitter_b.mean,
        )
