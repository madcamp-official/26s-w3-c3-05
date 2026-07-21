"""Two-phase look-to-register collection.

Phase 1 keeps the eyes on one center point while the user changes head/body pose.
Only these samples define the target center, MLP supervision, and 3D rays. Phase 2
keeps the head still while the eyes trace the four edges. Only these samples define
the target area. Keeping the two sets separate prevents boundary gaze from being
incorrectly labelled as the center direction during MLP training.
"""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np

from jarvis.calibration.registry import (
    TargetDirection,
    TargetGeometry3DRecord,
    TargetRecord,
    TargetSpread,
)
from jarvis.calibration.triangulation import TriangulationResult, triangulate_rays
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.direction import direction_to_yaw_pitch
from jarvis.gaze.feature_profile import (
    TargetFeatureSample,
    build_area_profile,
    build_feature_profile,
)
from jarvis.gaze.features import Vector3
from jarvis.gaze.smoothing import SmoothedGaze


class RegistrationPhase(StrEnum):
    CENTER = "CENTER"
    BOUNDARY = "BOUNDARY"
    COMPLETE = "COMPLETE"


class TargetRegistrationSession:
    def __init__(
        self,
        target_id: str,
        name: str,
        device_type: str,
        device_id: str,
        *,
        center_duration_ms: int = 20_000,
        boundary_duration_ms: int = 16_000,
        minimum_valid_frames: int = 30,
        minimum_boundary_frames: int | None = None,
        minimum_confidence: float = 0.35,
        maximum_jump_deg: float = 18.0,
        config: GazeConfig = GazeConfig(),
    ) -> None:
        if center_duration_ms <= 0 or boundary_duration_ms <= 0 or minimum_valid_frames <= 0:
            raise ValueError("duration and frame count must be positive")
        if minimum_boundary_frames is not None and minimum_boundary_frames <= 0:
            raise ValueError("minimum boundary frame count must be positive")
        self.target_id, self.name = target_id, name
        self.device_type, self.device_id = device_type, device_id
        self.center_duration_ms = center_duration_ms
        self.boundary_duration_ms = boundary_duration_ms
        self.duration_ms = center_duration_ms + boundary_duration_ms
        self.minimum_valid_frames = minimum_valid_frames
        self.minimum_boundary_frames = minimum_boundary_frames or minimum_valid_frames
        self.minimum_confidence, self.maximum_jump_deg = minimum_confidence, maximum_jump_deg
        self.config = config
        self.phase = RegistrationPhase.CENTER
        self.started_at_ms: int | None = None
        self.phase_started_at_ms: int | None = None
        self._center_samples: list[tuple[float, float]] = []
        self._boundary_samples: list[tuple[float, float]] = []
        self._center_rays: list[tuple[Vector3, Vector3]] = []
        self._center_face_scales: list[float] = []
        self._center_feature_samples: list[TargetFeatureSample] = []
        self._boundary_feature_samples: list[TargetFeatureSample] = []
        self._calibration_features: list[tuple[float, ...]] = []
        self.total_frames_seen = 0
        self.rejected_tracking_lost = 0
        self.rejected_closed_eyes = 0
        self.rejected_low_confidence = 0
        self.rejected_jump = 0
        self.triangulation_result: TriangulationResult | None = None
        """가장 최근 `finalize()` 호출이 시도한 삼각측량 결과 — 품질 기준을
        만족하지 못해 각도 모드로 대체된 경우에도 진단을 위해 남는다(값을
        숨기지 않는다). 광선이 아예 부족해 시도조차 못 했으면 None이다."""

    @property
    def valid_frame_count(self) -> int:
        return self.center_valid_frame_count + self.boundary_valid_frame_count

    @property
    def center_valid_frame_count(self) -> int:
        return len(self._center_samples)

    @property
    def boundary_valid_frame_count(self) -> int:
        return len(self._boundary_samples)

    @property
    def calibration_features(self) -> tuple[tuple[float, ...], ...]:
        """Raw regression features from CENTER only."""
        return tuple(self._calibration_features)

    @property
    def center_yaw_pitch(self) -> tuple[float, float] | None:
        if self.center_valid_frame_count < self.minimum_valid_frames:
            return None
        center = np.median(np.asarray(self._center_samples, dtype=np.float64), axis=0)
        return float(center[0]), float(center[1])

    @property
    def feature_samples(self) -> tuple[TargetFeatureSample, ...]:
        """Center and boundary evidence used by the target classifier."""
        return tuple((*self._center_feature_samples, *self._boundary_feature_samples))

    def phase_elapsed_ms(self, timestamp_ms: int) -> int:
        if self.phase_started_at_ms is None:
            return 0
        return max(0, timestamp_ms - self.phase_started_at_ms)

    def phase_duration_ms(self) -> int:
        if self.phase == RegistrationPhase.CENTER:
            return self.center_duration_ms
        if self.phase == RegistrationPhase.BOUNDARY:
            return self.boundary_duration_ms
        return 0

    def phase_progress(self, timestamp_ms: int) -> float:
        if self.phase == RegistrationPhase.COMPLETE:
            return 1.0
        duration = self.phase_duration_ms()
        time_progress = min(1.0, self.phase_elapsed_ms(timestamp_ms) / duration)
        if self.phase == RegistrationPhase.CENTER:
            frame_progress = min(1.0, self.center_valid_frame_count / self.minimum_valid_frames)
        else:
            frame_progress = min(
                1.0, self.boundary_valid_frame_count / self.minimum_boundary_frames
            )
        return min(time_progress, frame_progress)

    def add(
        self,
        gaze: SmoothedGaze | None,
        confidence: float,
        *,
        eyes_open: bool = True,
        face_scale: float | None = None,
        feature_sample: TargetFeatureSample | None = None,
        calibration_features: tuple[float, ...] | None = None,
    ) -> bool:
        self.total_frames_seen += 1
        if gaze is not None:
            self._advance_phase_if_ready(gaze.timestamp_ms)
        if self.phase == RegistrationPhase.COMPLETE:
            return False
        if gaze is None:
            self.rejected_tracking_lost += 1
            return False
        if not eyes_open:
            self.rejected_closed_eyes += 1
            return False
        if confidence < self.minimum_confidence:
            self.rejected_low_confidence += 1
            return False
        if self.started_at_ms is None:
            self.started_at_ms = gaze.timestamp_ms
            self.phase_started_at_ms = gaze.timestamp_ms
        accepted_phase = self.phase
        yaw, pitch = direction_to_yaw_pitch(gaze.direction)
        samples = (
            self._center_samples
            if accepted_phase == RegistrationPhase.CENTER
            else self._boundary_samples
        )
        if samples:
            previous_yaw, previous_pitch = samples[-1]
            if math.hypot(yaw - previous_yaw, pitch - previous_pitch) > self.maximum_jump_deg:
                self.rejected_jump += 1
                return False
        samples.append((yaw, pitch))
        if accepted_phase == RegistrationPhase.CENTER:
            if gaze.origin is not None:
                self._center_rays.append((gaze.origin, gaze.direction))
            if face_scale is not None and math.isfinite(face_scale) and face_scale > 0.0:
                self._center_face_scales.append(face_scale)
            if feature_sample is not None:
                self._center_feature_samples.append(feature_sample)
            if calibration_features is not None:
                self._calibration_features.append(calibration_features)
        elif feature_sample is not None:
            self._boundary_feature_samples.append(feature_sample)
        self._advance_phase_if_ready(gaze.timestamp_ms)
        return True

    def is_elapsed(self, timestamp_ms: int) -> bool:
        self._advance_phase_if_ready(timestamp_ms)
        return self.phase == RegistrationPhase.COMPLETE

    def start_boundary(self, timestamp_ms: int) -> None:
        """Advance explicitly after enough center frames; useful for UI/tests."""
        if self.center_valid_frame_count < self.minimum_valid_frames:
            raise ValueError(
                "not enough valid center frames: "
                f"{self.center_valid_frame_count}/{self.minimum_valid_frames}"
            )
        self.phase = RegistrationPhase.BOUNDARY
        self.phase_started_at_ms = timestamp_ms

    def _advance_phase_if_ready(self, timestamp_ms: int) -> None:
        if self.phase_started_at_ms is None:
            return
        elapsed_ms = self.phase_elapsed_ms(timestamp_ms)
        if (
            self.phase == RegistrationPhase.CENTER
            and elapsed_ms >= self.center_duration_ms
            and self.center_valid_frame_count >= self.minimum_valid_frames
        ):
            self.start_boundary(timestamp_ms)
            return
        if (
            self.phase == RegistrationPhase.BOUNDARY
            and elapsed_ms >= self.boundary_duration_ms
            and self.boundary_valid_frame_count >= self.minimum_boundary_frames
        ):
            self.phase = RegistrationPhase.COMPLETE
            self.phase_started_at_ms = timestamp_ms

    def finalize(self) -> TargetRecord:
        if self.center_valid_frame_count < self.minimum_valid_frames:
            raise ValueError(
                "not enough valid center frames: "
                f"{self.center_valid_frame_count}/{self.minimum_valid_frames}"
            )
        if self.boundary_valid_frame_count < self.minimum_boundary_frames:
            raise ValueError(
                "not enough valid boundary frames: "
                f"{self.boundary_valid_frame_count}/{self.minimum_boundary_frames}"
            )
        boundary_samples = np.asarray(self._boundary_samples, dtype=np.float64)
        center_yaw_pitch = self.center_yaw_pitch
        assert center_yaw_pitch is not None
        center = np.asarray(center_yaw_pitch, dtype=np.float64)
        deviations = np.abs(boundary_samples - center)
        spread = np.percentile(deviations, 90, axis=0)
        spread_yaw = min(
            self.config.registration_max_spread_deg,
            max(self.config.registration_min_spread_deg, float(spread[0])),
        )
        spread_pitch = min(
            self.config.registration_max_spread_deg,
            max(self.config.registration_min_spread_deg, float(spread[1])),
        )
        position_3d = self._try_triangulate()
        if self.config.require_3d_target_registration and position_3d is None:
            raise ValueError(f"3D target registration failed: {self.triangulation_diagnostic()}")
        feature_profile = (
            build_feature_profile(list(self.feature_samples)).profile
            if len(self.feature_samples) >= self.minimum_valid_frames
            else None
        )
        area_profile = build_area_profile(
            self._boundary_samples,
            center_yaw_pitch=(float(center[0]), float(center[1])),
            minimum_radius_deg=self.config.registration_min_spread_deg,
            maximum_radius_deg=self.config.registration_max_area_radius_deg,
        )
        return TargetRecord(
            target_id=self.target_id,
            name=self.name,
            device_type=self.device_type,
            direction=TargetDirection(float(center[0]), float(center[1])),
            spread=TargetSpread(spread_yaw, spread_pitch),
            device_id=self.device_id,
            position_3d=position_3d,
            reference_face_scale=(
                float(np.median(np.asarray(self._center_face_scales, dtype=np.float64)))
                if self._center_face_scales
                else None
            ),
            feature_profile=feature_profile,
            area_profile=area_profile,
        )

    def _try_triangulate(self) -> TargetGeometry3DRecord | None:
        """가능하면 3D 위치를 추정하고, 품질 기준을 만족할 때만 반환한다.

        기준 미달(머리 이동 부족, 광선이 거의 평행, 잔차 과다)이면 조용히
        None을 반환해 각도 기반 등록으로 대체되게 한다 — 대신 진단 결과는
        `self.triangulation_result`에 남겨 호출자가 왜 대체됐는지 로그로 보여줄
        수 있게 한다(지어낸 성공을 반환하지 않는다).
        """
        if len(self._center_rays) < self.config.minimum_triangulation_frames:
            self.triangulation_result = None
            return None
        origins = [origin for origin, _ in self._center_rays]
        directions = [direction for _, direction in self._center_rays]
        result = triangulate_rays(origins, directions)
        self.triangulation_result = result
        if not result.passes_quality_gates(self.config):
            return None
        radius_mm = max(result.residual_rms_mm, self.config.target_radius_floor_mm)
        center = result.center_mm
        return TargetGeometry3DRecord(
            center_mm=(float(center[0]), float(center[1]), float(center[2])),
            radius_mm=radius_mm,
        )

    def diagnostic_summary(self) -> str:
        """Human-readable counts explaining why registration frames were rejected."""
        return (
            f"phase={self.phase}, seen={self.total_frames_seen}, "
            f"center={self.center_valid_frame_count}, boundary={self.boundary_valid_frame_count}, "
            f"center_rays={len(self._center_rays)}, "
            f"center_scale={len(self._center_face_scales)}, "
            f"center_features={len(self._center_feature_samples)}, "
            f"boundary_features={len(self._boundary_feature_samples)}, "
            f"mlp_features={len(self._calibration_features)}, "
            f"tracking_lost={self.rejected_tracking_lost}, "
            f"closed_eyes={self.rejected_closed_eyes}, "
            f"low_conf={self.rejected_low_confidence}, jump={self.rejected_jump}"
        )

    def triangulation_diagnostic(self) -> str:
        """Human-readable 3D registration quality report."""
        if len(self._center_rays) < self.config.minimum_triangulation_frames:
            return (
                f"not enough center gaze rays: {len(self._center_rays)}/"
                f"{self.config.minimum_triangulation_frames}"
            )
        result = self.triangulation_result
        if result is None:
            return "triangulation was not attempted"
        checks = [
            (
                "baseline",
                result.baseline_mm,
                self.config.minimum_triangulation_baseline_mm,
                result.baseline_mm >= self.config.minimum_triangulation_baseline_mm,
            ),
            (
                "eigen",
                result.min_eigenvalue,
                self.config.minimum_triangulation_eigenvalue,
                result.min_eigenvalue >= self.config.minimum_triangulation_eigenvalue,
            ),
            (
                "residual",
                result.residual_rms_mm,
                self.config.maximum_triangulation_residual_mm,
                result.residual_rms_mm <= self.config.maximum_triangulation_residual_mm,
            ),
        ]
        failed = [
            f"{name}={value:.3f} threshold={threshold:.3f}"
            for name, value, threshold, passed in checks
            if not passed
        ]
        status = "ok" if not failed else "failed " + ", ".join(failed)
        return (
            f"{status}; rays={result.frame_count}, baseline={result.baseline_mm:.1f}mm, "
            f"residual={result.residual_rms_mm:.1f}mm, eigen={result.min_eigenvalue:.4f}"
        )
