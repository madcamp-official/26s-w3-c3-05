"""Fixation-sweep diagnostics for the gaze composition formula.

한 물리적 지점을 계속 응시하면서 고개만 돌리는 스윕에서, 합성식

    final_yaw = (head_yaw * head_yaw_weight + iris_x * max_eye_offset_deg) * sign

이 자세 불변이려면 `d(iris_x)/d(head_yaw) = -head_yaw_weight / max_eye_offset_deg`
이어야 한다. 따라서 스윕 데이터에서 iris↔head 회귀 기울기를 구하면 이 사용자·
카메라 조합에 맞는 암시 가중치(implied weight)가 닫힌 형태로 나온다:

    implied_weight = -max_eye_offset_deg * slope

같은 캡처에서 iris offset이 `max_valid_eye_offset` 클램프에 걸리는 비율도 함께
계산한다 — 큰 head yaw에서 눈 보상이 클램프에 잘려 프레임이 통째로 거부되고
있는지(documents/gaze.md 2026-07-21 진단의 후보 원인) 확인하기 위해서다.

이 모듈은 순수 `FaceObservation` 값만 다루므로 카메라·모델 파일 없이 단위
테스트한다. 실카메라 캡처는 `jarvis-gaze diagnose-composition`(cli.py)이 맡는다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jarvis.gaze.config import GazeConfig
from jarvis.gaze.features import FaceObservation, _rotate_2d


@dataclass(frozen=True, slots=True)
class AxisFit:
    """한 축(yaw 또는 pitch)에 대한 iris↔head 회귀 결과."""

    slope_offset_per_deg: float
    """head 각도 1도당 (roll 보정된) 평균 iris offset 변화량."""

    intercept_offset: float
    r_squared: float
    """0에 가까우면 잡음/비선형이 지배적이라 implied_weight를 믿을 수 없다."""

    implied_weight: float
    """자세 불변이 되기 위한 head weight: -max_eye_offset_deg * slope."""

    current_weight: float
    composed_std_current_deg: float
    """현재 가중치로 합성했을 때 응시 중 합성 각도의 표준편차(이상적으로 0)."""

    composed_std_implied_deg: float
    """implied_weight로 합성했을 때의 표준편차 — current와의 차이가 개선 폭이다."""

    head_range_deg: float
    """스윕이 실제로 커버한 head 각도 범위(max-min). 좁으면 회귀가 무의미하다."""

    frame_count: int


@dataclass(frozen=True, slots=True)
class ClampStats:
    """gaze_probe와 동일한 기준(양안 평균 offset의 max(|x|,|y|))의 클램프 통계."""

    limit: float
    rejected_fraction: float
    """max_valid_eye_offset 초과로 런타임이 프레임을 거부했을 비율."""

    near_clamp_fraction: float
    """클램프의 near_clamp_ratio 배 초과 ~ 클램프 이하 구간 비율(포화 임박)."""

    max_abs_offset: float


@dataclass(frozen=True, slots=True)
class CompositionDiagnostics:
    total_frames: int
    valid_frames: int
    """face 검출 + tracking confidence 충족 + 눈 뜬 프레임 수."""

    clamp: ClampStats | None
    yaw: AxisFit | None
    """head yaw가 거의 움직이지 않은 캡처면 None(회귀 불가 — 지어내지 않는다)."""

    pitch: AxisFit | None


_MINIMUM_HEAD_VARIANCE = 1e-6
"""head 각도가 사실상 상수인 축은 회귀 대상에서 제외하는 분산 하한."""


def _axis_fit(
    head_deg: npt.NDArray[np.float64],
    offsets: npt.NDArray[np.float64],
    current_weight: float,
    max_eye_offset_deg: float,
) -> AxisFit | None:
    head_variance = float(np.var(head_deg))
    if head_variance < _MINIMUM_HEAD_VARIANCE:
        return None
    head_mean = float(np.mean(head_deg))
    offset_mean = float(np.mean(offsets))
    covariance = float(np.mean((head_deg - head_mean) * (offsets - offset_mean)))
    slope = covariance / head_variance
    intercept = offset_mean - slope * head_mean
    offset_variance = float(np.var(offsets))
    r_squared = (
        (covariance * covariance) / (head_variance * offset_variance)
        if offset_variance > 0.0
        else 0.0
    )
    implied_weight = -max_eye_offset_deg * slope

    def composed_std(weight: float) -> float:
        composed = head_deg * weight + offsets * max_eye_offset_deg
        return float(np.std(composed))

    return AxisFit(
        slope_offset_per_deg=slope,
        intercept_offset=intercept,
        r_squared=max(0.0, min(1.0, r_squared)),
        implied_weight=implied_weight,
        current_weight=current_weight,
        composed_std_current_deg=composed_std(current_weight),
        composed_std_implied_deg=composed_std(implied_weight),
        head_range_deg=float(np.max(head_deg) - np.min(head_deg)),
        frame_count=int(head_deg.shape[0]),
    )


def analyze_fixation_sweep(
    observations: Sequence[FaceObservation],
    config: GazeConfig = GazeConfig(),
    *,
    near_clamp_ratio: float = 0.85,
) -> CompositionDiagnostics:
    """응시 고정 + 고개 스윕 캡처를 분석한다.

    유효 프레임 기준은 compose_gaze_vector와 동일하다(face 검출, tracking
    confidence, 눈 뜸). 회귀에는 클램프 초과 프레임도 포함한다 — 제외하면
    포화 구간에서 기울기가 낙관적으로 왜곡된다.
    """
    if not 0.0 < near_clamp_ratio < 1.0:
        raise ValueError("near_clamp_ratio must be within (0, 1)")

    valid: list[FaceObservation] = [
        observation
        for observation in observations
        if observation.face_detected
        and observation.eyes_open
        and min(observation.eye_tracking_confidence, observation.face_tracking_confidence)
        >= config.minimum_tracking_confidence
    ]
    if not valid:
        return CompositionDiagnostics(
            total_frames=len(observations),
            valid_frames=0,
            clamp=None,
            yaw=None,
            pitch=None,
        )

    mean_x = np.array(
        [(o.left_iris_relative[0] + o.right_iris_relative[0]) / 2.0 for o in valid],
        dtype=np.float64,
    )
    mean_y = np.array(
        [(o.left_iris_relative[1] + o.right_iris_relative[1]) / 2.0 for o in valid],
        dtype=np.float64,
    )

    # 클램프 통계는 런타임 거부 기준(gaze_probe._maybe_reject_iris)과 동일하게
    # roll 보정 전 원시 평균 offset으로 계산한다.
    clamp_metric = np.maximum(np.abs(mean_x), np.abs(mean_y))
    limit = config.max_valid_eye_offset
    near_band = float(np.mean((clamp_metric > near_clamp_ratio * limit) & (clamp_metric <= limit)))
    clamp = ClampStats(
        limit=limit,
        rejected_fraction=float(np.mean(clamp_metric > limit)),
        near_clamp_fraction=near_band,
        max_abs_offset=float(np.max(clamp_metric)),
    )

    # 회귀는 합성식이 실제로 쓰는 값(roll 보정된 offset)으로 한다.
    rotated = np.array(
        [
            _rotate_2d(x, y, -observation.head_roll_deg)
            for x, y, observation in zip(mean_x, mean_y, valid, strict=True)
        ],
        dtype=np.float64,
    )
    head_yaw = np.array([o.head_yaw_deg for o in valid], dtype=np.float64)
    head_pitch = np.array([o.head_pitch_deg for o in valid], dtype=np.float64)

    return CompositionDiagnostics(
        total_frames=len(observations),
        valid_frames=len(valid),
        clamp=clamp,
        yaw=_axis_fit(head_yaw, rotated[:, 0], config.head_yaw_weight, config.max_eye_offset_deg),
        pitch=_axis_fit(
            head_pitch, rotated[:, 1], config.head_pitch_weight, config.max_eye_offset_deg
        ),
    )


def summarize(diagnostics: CompositionDiagnostics) -> list[str]:
    """리포트 숫자를 실측 판단 문장으로 바꾼다(문서의 증상 가이드와 같은 톤)."""
    lines: list[str] = []
    if diagnostics.valid_frames == 0:
        lines.append("유효 프레임이 없습니다 — 조명/카메라/얼굴 가림을 확인하세요.")
        return lines
    clamp = diagnostics.clamp
    if clamp is not None:
        if clamp.rejected_fraction > 0.10:
            lines.append(
                f"iris offset이 클램프({clamp.limit:.2f})를 넘어 거부된 프레임이 "
                f"{clamp.rejected_fraction:.0%}입니다 — 클램프 포화가 실제로 발생하고 "
                "있습니다(거부 대신 신뢰도 하향 검토 근거)."
            )
        elif clamp.near_clamp_fraction > 0.20:
            lines.append(
                f"클램프 임박 구간 프레임이 {clamp.near_clamp_fraction:.0%}입니다 — "
                "스윕 범위 끝에서 보상이 한계에 가깝습니다."
            )
        else:
            lines.append("클램프 포화는 관측되지 않았습니다 — 주범이 아닐 가능성이 높습니다.")
    for axis_name, fit in (("yaw", diagnostics.yaw), ("pitch", diagnostics.pitch)):
        if fit is None:
            lines.append(f"{axis_name}: head 각도 변화가 없어 회귀를 계산하지 않았습니다.")
            continue
        if fit.head_range_deg < 15.0:
            lines.append(
                f"{axis_name}: 스윕 범위가 {fit.head_range_deg:.1f}도로 좁아 결과를 "
                "신뢰하기 어렵습니다 — 더 크게 고개를 돌려 다시 측정하세요."
            )
            continue
        if fit.r_squared < 0.5:
            lines.append(
                f"{axis_name}: R²={fit.r_squared:.2f}로 선형 적합이 약합니다 — 가중치 "
                "조정만으로는 부족하고 비선형(자세별) 보정이 필요하다는 신호입니다."
            )
            continue
        lines.append(
            f"{axis_name}: implied weight {fit.implied_weight:.2f} (현재 "
            f"{fit.current_weight:.2f}, R²={fit.r_squared:.2f}) — 합성 각도 흔들림 "
            f"{fit.composed_std_current_deg:.1f}° → {fit.composed_std_implied_deg:.1f}°."
        )
    return lines
