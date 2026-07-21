"""Target-specific feature profiles for gaze/head registration.

The classic gaze target matcher stores one direction plus one angular spread per
target.  That is easy to tune, but it collapses useful registration evidence
away: head pose, gaze direction, and camera distance all vary together when a
person looks at the same physical object from different poses.

This module keeps that evidence as a small statistical profile and scores live
frames with Mahalanobis distance.  It is intentionally lightweight: no offline
training job, no image storage, only numeric frame features collected during
look-to-register.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt


FEATURE_NAMES: tuple[str, ...] = (
    "gaze_yaw",
    "gaze_pitch",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "face_scale",
    "face_center_x",
    "face_center_y",
)
FEATURE_DIMENSION = len(FEATURE_NAMES)
HEAD_FEATURE_INDICES: tuple[int, ...] = (2, 3, 4)
"""Feature indices for head yaw/pitch/roll.

Head pose is useful context, but it should not veto a target when the final gaze
direction is close.  We therefore make head dimensions intentionally tolerant in
the learned distribution.
"""

GAZE_FEATURE_INDICES: tuple[int, ...] = (0, 1)
FACE_CONTEXT_INDICES: tuple[int, ...] = (5, 6, 7)

# Each signal uses different units. These standard-deviation floors prevent a
# nearly constant registration feature from dominating the inverse covariance,
# while keeping gaze more discriminative than head pose. Face scale and center
# stay meaningful instead of being erased by a degree-sized global regularizer.
FEATURE_STD_FLOORS = np.asarray(
    [1.5, 1.5, 6.0, 6.0, 8.0, 0.008, 0.035, 0.035],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class TargetFeatureSample:
    gaze_yaw: float
    gaze_pitch: float
    head_yaw: float
    head_pitch: float
    head_roll: float
    face_scale: float
    face_center_x: float = 0.5
    face_center_y: float = 0.5

    def as_array(self) -> npt.NDArray[np.float64]:
        values = np.array(
            [
                self.gaze_yaw,
                self.gaze_pitch,
                self.head_yaw,
                self.head_pitch,
                self.head_roll,
                self.face_scale,
                self.face_center_x,
                self.face_center_y,
            ],
            dtype=np.float64,
        )
        if values.shape != (FEATURE_DIMENSION,) or not np.all(np.isfinite(values)):
            raise ValueError("target feature sample must contain finite values")
        if self.face_scale <= 0.0:
            raise ValueError("face_scale must be positive")
        if not 0.0 <= self.face_center_x <= 1.0 or not 0.0 <= self.face_center_y <= 1.0:
            raise ValueError("face center must be normalized within [0, 1]")
        return values

    @classmethod
    def from_array(cls, values: npt.NDArray[np.float64]) -> TargetFeatureSample:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (FEATURE_DIMENSION,) or not np.all(np.isfinite(array)):
            raise ValueError("target feature sample must contain finite values")
        return cls(
            gaze_yaw=float(array[0]),
            gaze_pitch=float(array[1]),
            head_yaw=float(array[2]),
            head_pitch=float(array[3]),
            head_roll=float(array[4]),
            face_scale=float(array[5]),
            face_center_x=float(array[6]),
            face_center_y=float(array[7]),
        )


@dataclass(frozen=True, slots=True)
class TargetFeatureProfile:
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    sample_count: int
    threshold: float

    def __post_init__(self) -> None:
        mean = self.mean_array
        covariance = self.covariance_array
        if mean.shape != (FEATURE_DIMENSION,):
            raise ValueError("feature profile mean has invalid shape")
        if covariance.shape != (FEATURE_DIMENSION, FEATURE_DIMENSION):
            raise ValueError("feature profile covariance has invalid shape")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("feature profile values must be finite")
        if self.sample_count <= 0:
            raise ValueError("feature profile sample_count must be positive")
        if not math.isfinite(self.threshold) or self.threshold <= 0.0:
            raise ValueError("feature profile threshold must be finite and positive")

    @property
    def mean_array(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.mean, dtype=np.float64)

    @property
    def covariance_array(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.covariance, dtype=np.float64)

    @property
    def inverse_covariance(self) -> npt.NDArray[np.float64]:
        return _mahalanobis_operands(self)[1]

    def mahalanobis_distance(self, sample: TargetFeatureSample) -> float:
        mean, inverse_covariance = _mahalanobis_operands(self)
        delta = sample.as_array() - mean
        distance_sq = float(delta.T @ inverse_covariance @ delta)
        return math.sqrt(max(0.0, distance_sq))


@lru_cache(maxsize=256)
def _mahalanobis_operands(
    profile: TargetFeatureProfile,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Cache the mean vector and inverse covariance per immutable profile.

    The profile is a frozen, hashable dataclass, so this pair is computed only
    once per registration. Returned arrays are read-only so callers cannot
    corrupt the shared cache.
    """
    mean = np.asarray(profile.mean, dtype=np.float64)
    inverse = _invert_covariance(profile.covariance)
    mean.setflags(write=False)
    inverse.setflags(write=False)
    return mean, inverse


def _invert_covariance(
    covariance: tuple[tuple[float, ...], ...],
) -> npt.NDArray[np.float64]:
    """Invert the fixed 8D covariance without a native LAPACK call.

    New profiles are positive definite because every feature receives a
    variance floor. Pivoting Gauss-Jordan is stable enough for this tiny matrix
    and avoids a Windows NumPy/MKL crash when Torch has loaded another OpenMP
    runtime. A malformed singular legacy matrix falls back to diagonal scoring.
    """
    size = FEATURE_DIMENSION
    matrix = [
        [float(value) for value in row]
        + [1.0 if row_index == column else 0.0 for column in range(size)]
        for row_index, row in enumerate(covariance)
    ]
    scale = max(abs(value) for row in covariance for value in row)
    pivot_floor = max(1e-12, scale * 1e-12)

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot_row][column]) <= pivot_floor:
            diagonal = np.zeros((size, size), dtype=np.float64)
            for index in range(size):
                variance = max(
                    abs(float(covariance[index][index])),
                    float(FEATURE_STD_FLOORS[index] ** 2),
                )
                diagonal[index, index] = 1.0 / variance
            return diagonal
        if pivot_row != column:
            matrix[column], matrix[pivot_row] = matrix[pivot_row], matrix[column]

        pivot = matrix[column][column]
        matrix[column] = [value / pivot for value in matrix[column]]
        for row in range(size):
            if row == column:
                continue
            factor = matrix[row][column]
            if factor == 0.0:
                continue
            matrix[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(matrix[row], matrix[column], strict=True)
            ]

    return np.asarray([row[size:] for row in matrix], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class FeatureProfileBuildResult:
    profile: TargetFeatureProfile
    kept_samples: int
    rejected_outliers: int


@dataclass(frozen=True, slots=True)
class TargetAreaProfile:
    center_yaw: float
    center_pitch: float
    radius_yaw: float
    radius_pitch: float
    sample_count: int

    def __post_init__(self) -> None:
        values = (self.center_yaw, self.center_pitch, self.radius_yaw, self.radius_pitch)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("target area profile values must be finite")
        if self.radius_yaw <= 0.0 or self.radius_pitch <= 0.0:
            raise ValueError("target area profile radii must be positive")
        if self.sample_count <= 0:
            raise ValueError("target area profile sample_count must be positive")

    def normalized_distance(
        self,
        gaze_yaw: float,
        gaze_pitch: float,
        max_radius_deg: float | None = None,
        radius_scale: float = 1.0,
    ) -> float:
        radius_yaw = (
            min(self.radius_yaw, max_radius_deg)
            if max_radius_deg is not None
            else self.radius_yaw
        )
        radius_pitch = (
            min(self.radius_pitch, max_radius_deg) if max_radius_deg is not None else self.radius_pitch
        )
        radius_yaw *= radius_scale
        radius_pitch *= radius_scale
        return math.hypot(
            (gaze_yaw - self.center_yaw) / radius_yaw,
            (gaze_pitch - self.center_pitch) / radius_pitch,
        )

    def contains(
        self,
        gaze_yaw: float,
        gaze_pitch: float,
        max_radius_deg: float | None = None,
        radius_scale: float = 1.0,
    ) -> bool:
        return self.normalized_distance(
            gaze_yaw,
            gaze_pitch,
            max_radius_deg,
            radius_scale,
        ) <= 1.0


def build_area_profile(
    yaw_pitch_samples: list[tuple[float, float]],
    *,
    center_yaw_pitch: tuple[float, float] | None = None,
    minimum_radius_deg: float = 3.0,
    maximum_radius_deg: float | None = None,
    padding_scale: float = 1.0,
) -> TargetAreaProfile:
    if not yaw_pitch_samples:
        raise ValueError("at least one yaw/pitch sample is required")
    matrix = np.asarray(yaw_pitch_samples, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("yaw/pitch samples must be finite pairs")
    center = (
        np.asarray(center_yaw_pitch, dtype=np.float64)
        if center_yaw_pitch is not None
        else np.median(matrix, axis=0)
    )
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("area center must be a finite yaw/pitch pair")
    deviations = np.abs(matrix - center)
    radius = np.percentile(deviations, 95, axis=0) * padding_scale
    radius = np.maximum(radius, minimum_radius_deg)
    if maximum_radius_deg is not None:
        radius = np.minimum(radius, maximum_radius_deg)
    return TargetAreaProfile(
        center_yaw=float(center[0]),
        center_pitch=float(center[1]),
        radius_yaw=float(radius[0]),
        radius_pitch=float(radius[1]),
        sample_count=int(len(matrix)),
    )


def build_feature_profile(
    samples: list[TargetFeatureSample],
    *,
    regularization: float = 1e-6,
    threshold_floor: float = 2.5,
    threshold_quantile: float = 0.90,
) -> FeatureProfileBuildResult:
    """Build a robust target feature distribution.

    The vector mixes degrees and normalized image coordinates. Per-feature
    variance floors establish the intended balance; `regularization` only adds
    numerical stability.
    """
    if not samples:
        raise ValueError("at least one feature sample is required")
    matrix = np.asarray([sample.as_array() for sample in samples], dtype=np.float64)
    center = np.median(matrix, axis=0)
    deviations = np.abs(matrix - center)
    mad = np.median(deviations, axis=0)
    robust_scale = np.maximum(mad * 1.4826, FEATURE_STD_FLOORS)
    normalized = deviations / robust_scale
    keep_mask = np.max(normalized, axis=1) <= 4.0
    kept = matrix[keep_mask]
    if len(kept) < max(3, min(len(matrix), FEATURE_DIMENSION)):
        kept = matrix
        rejected = 0
    else:
        rejected = int(len(matrix) - len(kept))

    mean = np.mean(kept, axis=0)
    if len(kept) == 1:
        covariance = np.eye(FEATURE_DIMENSION, dtype=np.float64)
    else:
        # ``np.cov`` can abort the Windows process when NumPy/MKL and Torch's
        # OpenMP runtime have both been loaded.  This is the same unbiased
        # sample-covariance formula, expressed directly for our tiny 8D
        # matrix, and avoids that separate native code path.
        centered = kept - mean
        covariance = np.einsum("ni,nj->ij", centered, centered, optimize=False)
        covariance /= float(len(kept) - 1)
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance += np.diag(FEATURE_STD_FLOORS**2)
    covariance += np.eye(FEATURE_DIMENSION, dtype=np.float64) * regularization
    provisional = TargetFeatureProfile(
        mean=tuple(float(value) for value in mean),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        sample_count=int(len(kept)),
        threshold=threshold_floor,
    )
    distances = np.asarray(
        [
            provisional.mahalanobis_distance(TargetFeatureSample.from_array(row))
            for row in kept
        ],
        dtype=np.float64,
    )
    threshold = max(threshold_floor, float(np.quantile(distances, threshold_quantile)) * 1.15)
    profile = TargetFeatureProfile(
        mean=provisional.mean,
        covariance=provisional.covariance,
        sample_count=provisional.sample_count,
        threshold=threshold,
    )
    return FeatureProfileBuildResult(
        profile=profile,
        kept_samples=int(len(kept)),
        rejected_outliers=rejected,
    )
