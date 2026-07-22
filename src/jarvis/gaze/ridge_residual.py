"""Ridge residual gaze 보정의 오프라인 학습·A/B 평가 (런타임 미연결).

raw gaze의 오차(delta = target 중심 − raw gaze)를 8D feature로 회귀한다 —
절대 좌표를 직접 학습하면 모델이 항상 등록 물체 중심만 출력하므로 residual만
배운다. 이 저장소는 2026-07-21에 residual MLP·Ridge를 제거한 이력이 있고
(documents/decisions.md), 2026-07-22 실측에서 자세별 편향이 세션 간 요동함을
확인했다. 따라서 이 모듈은 **런타임에 연결되지 않는다** — 등록 시 저장한
원시 샘플(target_registration의 raw_sample_dir)로 leave-one-yaw-bin-out A/B를
돌려, 현재 bin 보정표 대비 held-out 오차가 실제로 줄 때만 활성화를 논의한다
(`jarvis-gaze ab-residual`).

인접 프레임을 무작위로 섞어 나누면 성능이 부풀려지므로 split은 반드시 자세
구간(head-yaw bin) 단위다. 행렬 해는 `np.linalg` 없이 부분 피벗 Gauss 소거로
구한다(이 머신의 NumPy/MKL 크래시 회피 — feature_profile._invert_covariance와
같은 이유).

입력 feature에서 raw gaze yaw/pitch는 **의도적으로 제외**한다: 단일 target
데이터에서는 정답이 `delta = center − gaze`라서 gaze를 입력에 넣으면 모델이
"항상 물체 중심을 출력"하는 상수 예측기로 붕괴하고, 관계가 전역적으로
성립하므로 leave-bin-out으로도 잡히지 않는다(2026-07-22 단위 테스트로 확인).
따라서 delta는 자세·문맥 6D(head yaw/pitch/roll, face scale/center)로만
예측한다 — bin 보정표의 연속 일반화판이며, gaze 입력 복원은 여러 방향의
target 데이터를 합쳐 학습할 때만 다시 검토한다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jarvis.gaze.config import GazeConfig
from jarvis.gaze.feature_profile import (
    FEATURE_DIMENSION,
    TargetFeatureSample,
    build_pose_correction,
)

Matrix = npt.NDArray[np.float64]

POSE_FEATURE_DIMENSION = FEATURE_DIMENSION - 2
"""raw gaze 2차원을 제외한 자세·문맥 feature 수 (모듈 docstring 참고)."""


def _pose_features(sample: TargetFeatureSample) -> Matrix:
    return sample.as_array()[2:]


def _solve_linear(matrix: Matrix, rhs: Matrix) -> Matrix:
    """부분 피벗 Gauss 소거로 `matrix @ x = rhs`를 푼다 (LAPACK 미사용)."""
    size = matrix.shape[0]
    augmented = np.concatenate([matrix.astype(np.float64), rhs.astype(np.float64)], axis=1)
    for column in range(size):
        pivot_row = column + int(np.argmax(np.abs(augmented[column:, column])))
        if abs(augmented[pivot_row, column]) < 1e-12:
            raise ValueError("ridge normal matrix is singular")
        if pivot_row != column:
            augmented[[column, pivot_row]] = augmented[[pivot_row, column]]
        augmented[column] = augmented[column] / augmented[column, column]
        for row in range(size):
            if row != column and augmented[row, column] != 0.0:
                augmented[row] = augmented[row] - augmented[row, column] * augmented[column]
    return np.asarray(augmented[:, size:], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class RidgeResidualModel:
    """표준화된 자세·문맥 6D feature → (delta_yaw, delta_pitch) 선형 사상."""

    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    """(POSE_FEATURE_DIMENSION + 1) x 2 — 마지막 행이 bias."""

    def predict_delta(self, sample: TargetFeatureSample) -> tuple[float, float]:
        features = (_pose_features(sample) - np.asarray(self.feature_means)) / np.asarray(
            self.feature_scales
        )
        weights = np.asarray(self.weights, dtype=np.float64)
        delta = np.einsum("i,ik->k", features, weights[:-1], optimize=False) + weights[-1]
        return float(delta[0]), float(delta[1])

    def corrected_gaze(self, sample: TargetFeatureSample) -> tuple[float, float]:
        delta_yaw, delta_pitch = self.predict_delta(sample)
        return sample.gaze_yaw + delta_yaw, sample.gaze_pitch + delta_pitch


def train_ridge_residual(
    samples: Sequence[TargetFeatureSample],
    center_yaw_pitch: tuple[float, float],
    *,
    ridge_lambda: float = 1.0,
) -> RidgeResidualModel:
    """중앙 응시 샘플로 residual Ridge를 닫힌 형태로 학습한다.

    정답은 `delta = center − raw_gaze`다. 한 물체의 데이터만 쓰면 그 물체
    주변에서만 유효하다 — 전역 보정으로 쓰려면 여러 방향의 target 데이터를
    합쳐야 한다(호출자 책임).
    """
    if len(samples) < POSE_FEATURE_DIMENSION + 2:
        raise ValueError(
            f"need at least {POSE_FEATURE_DIMENSION + 2} samples, got {len(samples)}"
        )
    if not math.isfinite(ridge_lambda) or ridge_lambda < 0.0:
        raise ValueError("ridge_lambda must be finite and non-negative")

    gaze = np.stack([sample.as_array()[:2] for sample in samples])
    features = np.stack([_pose_features(sample) for sample in samples])
    center = np.asarray(center_yaw_pitch, dtype=np.float64)
    targets = center[None, :] - gaze

    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales = np.where(scales < 1e-6, 1.0, scales)
    standardized = (features - means) / scales
    design = np.concatenate(
        [standardized, np.ones((standardized.shape[0], 1), dtype=np.float64)], axis=1
    )

    # ``@``(BLAS/MKL)는 Torch의 OpenMP 런타임과 함께 로드되면 이 머신에서
    # 프로세스를 abort시킨다 — feature_profile의 covariance 계산과 같은 이유로
    # einsum(optimize=False)을 쓴다.
    normal = np.einsum("ni,nj->ij", design, design, optimize=False)
    moment = np.einsum("ni,nk->ik", design, targets, optimize=False)
    regularizer = np.eye(POSE_FEATURE_DIMENSION + 1, dtype=np.float64) * ridge_lambda
    regularizer[-1, -1] = 0.0  # bias는 벌점을 주지 않는다.
    weights = _solve_linear(normal + regularizer, moment)
    return RidgeResidualModel(
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        weights=tuple(tuple(float(value) for value in row) for row in weights),
    )


@dataclass(frozen=True, slots=True)
class HeldOutBinResult:
    label: str
    frame_count: int
    raw_error_deg: float
    """보정 없이 raw gaze와 중심 사이 각거리 중앙값."""

    bin_table_error_deg: float | None
    """현재 방식(bin 보정표, rescue와 동일 오프셋)의 held-out 오차. 학습 bin이
    부족해 보정표가 안 만들어지면 None."""

    ridge_error_deg: float


@dataclass(frozen=True, slots=True)
class ResidualAbReport:
    bins: tuple[HeldOutBinResult, ...]
    ridge_lambda: float

    @property
    def verdict_lines(self) -> list[str]:
        if not self.bins:
            return ["평가할 bin이 없습니다 — coverage를 채운 등록 데이터가 필요합니다."]
        ridge_wins = sum(1 for b in self.bins if b.ridge_error_deg < b.raw_error_deg)
        table_wins = sum(
            1
            for b in self.bins
            if b.bin_table_error_deg is not None and b.bin_table_error_deg < b.raw_error_deg
        )
        lines = [
            f"held-out bin {len(self.bins)}개 중 Ridge가 raw보다 나은 bin: {ridge_wins}개, "
            f"bin 보정표가 raw보다 나은 bin: {table_wins}개.",
        ]
        if ridge_wins >= max(1, round(len(self.bins) * 0.7)):
            lines.append(
                "판정: Ridge가 대부분 bin에서 raw를 이깁니다 — 단, 이 수치는 같은 캡처"
                " 세션 안의 결과입니다. 세션 간 요동(2026-07-22 실측)이 진짜 관문이므로,"
                " 다른 세션의 raw 샘플로 한 번 더 확인한 뒤 활성화를 결정하세요."
            )
        else:
            lines.append(
                "판정: Ridge가 held-out에서도 이점을 못 보입니다 — 활성화하지 않습니다."
                " (bin 보정표 rescue 유지)"
            )
        return lines


def evaluate_leave_one_bin_out(
    samples: Sequence[TargetFeatureSample],
    center_yaw_pitch: tuple[float, float],
    config: GazeConfig = GazeConfig(),
    *,
    ridge_lambda: float = 1.0,
    minimum_bin_frames: int = 15,
) -> ResidualAbReport:
    """head-yaw bin 하나씩을 통째로 held-out으로 빼며 A/B를 돌린다.

    무작위 프레임 split은 인접 프레임 유출로 성능이 부풀려지므로 쓰지 않는다.
    각 fold에서 (a) raw, (b) 현재 bin 보정표(rescue 오프셋과 동일), (c) Ridge
    residual의 held-out 각오차 중앙값을 비교한다.
    """
    center = np.asarray(center_yaw_pitch, dtype=np.float64)
    edges = (-math.inf, *config.pose_correction_bin_edges_deg, math.inf)
    bins: list[list[TargetFeatureSample]] = [[] for _ in range(len(edges) - 1)]
    for sample in samples:
        for index, (lower, upper) in enumerate(zip(edges, edges[1:], strict=False)):
            if lower <= sample.head_yaw < upper:
                bins[index].append(sample)
                break

    results: list[HeldOutBinResult] = []
    for index, held_out in enumerate(bins):
        if len(held_out) < minimum_bin_frames:
            continue
        train = [sample for other, members in enumerate(bins) if other != index for sample in members]
        if len(train) < POSE_FEATURE_DIMENSION + 2:
            continue

        def angular_error(gaze_yaw: float, gaze_pitch: float) -> float:
            return math.hypot(gaze_yaw - float(center[0]), gaze_pitch - float(center[1]))

        raw_errors = [angular_error(s.gaze_yaw, s.gaze_pitch) for s in held_out]

        correction = build_pose_correction(
            train,
            center_yaw_pitch=(float(center[0]), float(center[1])),
            reference_head_yaw_deg=None,
            bin_edges_deg=config.pose_correction_bin_edges_deg,
            minimum_bin_samples=config.pose_correction_min_bin_samples,
            maximum_offset_deg=config.pose_correction_max_offset_deg,
        )
        table_errors: list[float] | None = None
        if correction is not None:
            table_errors = []
            for sample in held_out:
                offset_yaw, offset_pitch = correction.offset_for(sample.head_yaw)
                table_errors.append(
                    angular_error(sample.gaze_yaw - offset_yaw, sample.gaze_pitch - offset_pitch)
                )

        model = train_ridge_residual(
            train, (float(center[0]), float(center[1])), ridge_lambda=ridge_lambda
        )
        ridge_errors = [angular_error(*model.corrected_gaze(s)) for s in held_out]

        lower, upper = edges[index], edges[index + 1]
        left = "-inf" if math.isinf(lower) else f"{lower:+.0f}"
        right = "+inf" if math.isinf(upper) else f"{upper:+.0f}"
        results.append(
            HeldOutBinResult(
                label=f"[{left},{right})",
                frame_count=len(held_out),
                raw_error_deg=float(np.median(np.asarray(raw_errors))),
                bin_table_error_deg=(
                    float(np.median(np.asarray(table_errors))) if table_errors else None
                ),
                ridge_error_deg=float(np.median(np.asarray(ridge_errors))),
            )
        )
    return ResidualAbReport(bins=tuple(results), ridge_lambda=ridge_lambda)
