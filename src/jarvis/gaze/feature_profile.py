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
)
FEATURE_DIMENSION = len(FEATURE_NAMES)
HEAD_FEATURE_INDICES: tuple[int, ...] = (2, 3, 4)
"""Feature indices for head yaw/pitch/roll.

Head pose is useful context, but it should not veto a target when the final gaze
direction is close.  We therefore make head dimensions intentionally tolerant in
the learned distribution.
"""

GAZE_FEATURE_INDICES: tuple[int, ...] = (0, 1)


@dataclass(frozen=True, slots=True)
class TargetFeatureSample:
    gaze_yaw: float
    gaze_pitch: float
    head_yaw: float
    head_pitch: float
    head_roll: float
    face_scale: float

    def as_array(self) -> npt.NDArray[np.float64]:
        values = np.array(
            [
                self.gaze_yaw,
                self.gaze_pitch,
                self.head_yaw,
                self.head_pitch,
                self.head_roll,
                self.face_scale,
            ],
            dtype=np.float64,
        )
        if values.shape != (FEATURE_DIMENSION,) or not np.all(np.isfinite(values)):
            raise ValueError("target feature sample must contain finite values")
        if self.face_scale <= 0.0:
            raise ValueError("face_scale must be positive")
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
    """Cache the mean vector and pseudo-inverse covariance per (immutable) profile.

    `pinv` runs an SVD; recomputing it for every frame × every registered target
    dominated the whole pure pipeline (>50% of `evaluate()` time). The profile is
    a frozen, hashable dataclass, so the pair is computed once per registration.
    Returned arrays are shared — marked read-only so a caller cannot corrupt the
    cache in place.
    """
    mean = np.asarray(profile.mean, dtype=np.float64)
    inverse = np.linalg.pinv(np.asarray(profile.covariance, dtype=np.float64))
    mean.setflags(write=False)
    inverse.setflags(write=False)
    return mean, inverse


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
        self, gaze_yaw: float, gaze_pitch: float, max_radius_deg: float | None = None
    ) -> float:
        radius_yaw = (
            min(self.radius_yaw, max_radius_deg)
            if max_radius_deg is not None
            else self.radius_yaw
        )
        radius_pitch = (
            min(self.radius_pitch, max_radius_deg) if max_radius_deg is not None else self.radius_pitch
        )
        return math.hypot(
            (gaze_yaw - self.center_yaw) / radius_yaw,
            (gaze_pitch - self.center_pitch) / radius_pitch,
        )

    def contains(
        self, gaze_yaw: float, gaze_pitch: float, max_radius_deg: float | None = None
    ) -> bool:
        return self.normalized_distance(gaze_yaw, gaze_pitch, max_radius_deg) <= 1.0


def build_area_profile(
    yaw_pitch_samples: list[tuple[float, float]],
    *,
    minimum_radius_deg: float = 3.0,
    maximum_radius_deg: float | None = None,
    padding_scale: float = 1.0,
) -> TargetAreaProfile:
    if not yaw_pitch_samples:
        raise ValueError("at least one yaw/pitch sample is required")
    matrix = np.asarray(yaw_pitch_samples, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("yaw/pitch samples must be finite pairs")
    center = np.median(matrix, axis=0)
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
    regularization: float = 1.0,
    head_regularization: float = 64.0,
    threshold_floor: float = 2.5,
    threshold_quantile: float = 0.90,
) -> FeatureProfileBuildResult:
    """Build a robust target feature distribution.

    `regularization` is diagonal covariance padding.  The feature vector mixes
    degrees and normalized face scale, so a small positive diagonal keeps the
    inverse covariance stable even when the user barely moves during
    registration.
    """
    if not samples:
        raise ValueError("at least one feature sample is required")
    matrix = np.asarray([sample.as_array() for sample in samples], dtype=np.float64)
    center = np.median(matrix, axis=0)
    deviations = np.abs(matrix - center)
    mad = np.median(deviations, axis=0)
    robust_scale = np.maximum(mad * 1.4826, np.array([1.0, 1.0, 2.0, 2.0, 2.0, 0.01]))
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
        covariance = np.cov(kept, rowvar=False)
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance += np.eye(FEATURE_DIMENSION, dtype=np.float64) * regularization
    for index in HEAD_FEATURE_INDICES:
        covariance[index, index] += head_regularization
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
