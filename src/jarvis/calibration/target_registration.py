"""Two-phase boundary collection for deterministic target profiles."""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np

from jarvis.calibration.registry import (
    TargetDirection,
    TargetRecord,
    TargetSpread,
)
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.direction import direction_to_yaw_pitch
from jarvis.gaze.feature_profile import (
    TargetFeatureSample,
    build_area_profile,
    build_feature_profile,
)
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
        self._center_face_scales: list[float] = []
        self._center_feature_samples: list[TargetFeatureSample] = []
        self._boundary_feature_samples: list[TargetFeatureSample] = []
        self.total_frames_seen = 0
        self.rejected_tracking_lost = 0
        self.rejected_closed_eyes = 0
        self.rejected_low_confidence = 0
        self.rejected_jump = 0

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
    def center_yaw_pitch(self) -> tuple[float, float] | None:
        # The precision boundary is the authoritative object extent. Its robust
        # midpoint is therefore the target center. Phase-1 samples exist to
        # learn pose/distance/face-location context, not a regression label.
        samples = (
            self._boundary_samples
            if self.boundary_valid_frame_count >= self.minimum_boundary_frames
            else self._center_samples
        )
        if len(samples) < self.minimum_valid_frames:
            return None
        center = np.median(np.asarray(samples, dtype=np.float64), axis=0)
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
            if face_scale is not None and math.isfinite(face_scale) and face_scale > 0.0:
                self._center_face_scales.append(face_scale)
            if feature_sample is not None:
                self._center_feature_samples.append(feature_sample)
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
        # Boundary rays touch different surface points, so triangulating them as
        # if they met at one 3D point would fabricate a target position. The
        # deterministic profile instead uses face scale and image location.
        position_3d = None
        if self.config.require_3d_target_registration and position_3d is None:
            raise ValueError("3D registration is incompatible with boundary tracing")
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
    def diagnostic_summary(self) -> str:
        """Human-readable counts explaining why registration frames were rejected."""
        return (
            f"phase={self.phase}, seen={self.total_frames_seen}, "
            f"center={self.center_valid_frame_count}, boundary={self.boundary_valid_frame_count}, "
            f"center_scale={len(self._center_face_scales)}, "
            f"center_features={len(self._center_feature_samples)}, "
            f"boundary_features={len(self._boundary_feature_samples)}, "
            f"tracking_lost={self.rejected_tracking_lost}, "
            f"closed_eyes={self.rejected_closed_eyes}, "
            f"low_conf={self.rejected_low_confidence}, jump={self.rejected_jump}"
        )
