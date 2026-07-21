"""Run the real Gaze Targeting pipeline on frames and expose every intermediate.

This is the crux of the debugging tool: instead of only showing the webcam, it
feeds each frame through the *actual* engine steps (features → smoothing →
classifier → lock) and captures the value produced at each one, so a person can
watch the pipeline work and see exactly where it stops working.

Honesty (development-principles.md 1·2): the orchestration below mirrors
``jarvis.gaze.engine.GazeTargetingEngine.process`` step for step, so the
``TargetEstimate`` shown here is the same message the engine would emit — no
separate, drifting approximation. When tracking is lost, or no device profiles
are registered, or mediapipe/model is absent, that is surfaced as-is rather than
hidden behind a plausible-looking number.

Import safety: this module imports only the pure gaze code (features/smoothing/
classifier/lock/config), never mediapipe. The MediaPipe adapter is imported
lazily inside :meth:`GazeProbe.create`, so ``evaluate`` and the snapshot types
are unit-testable with synthetic observations and no ``vision`` extra installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from jarvis.contracts.messages import TargetEstimate
from jarvis.gaze.calibration_model import GazeCalibrationModel, observation_features
from jarvis.gaze.classifier import (
    ClassificationResult,
    DeviceGazeProfile,
    TargetClassifier,
    TargetGeometry3D,
    effective_distance_and_variance,
)
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.direction import direction_to_yaw_pitch
from jarvis.gaze.feature_profile import (
    TargetAreaProfile,
    TargetFeatureProfile,
    TargetFeatureSample,
)
from jarvis.gaze.features import FaceObservation, GazeVector, Vector3, compose_gaze_vector
from jarvis.gaze.lock import GazeLockStateMachine, GazeLockState
from jarvis.gaze.smoothing import GazeSmoother


@dataclass(frozen=True, slots=True)
class DeviceGazeDetail:
    """Per-registered-device diagnostic: how far the gaze points from it.

    ``angular_distance_deg`` is the angle between the (smoothed) gaze direction
    and this device's registered mean direction — a transparent geometric fact
    computed from public profiles, shown alongside the engine's authoritative
    classification.
    """

    device_id: str
    angular_distance_deg: float
    allowed_radius_deg: float
    normalized_distance: float
    target_yaw_deg: float
    target_pitch_deg: float
    within_profile_radius: bool
    is_selected: bool

    @property
    def range_status(self) -> str:
        """Human-readable live range check for the registered target profile."""
        if math.isnan(self.angular_distance_deg):
            return "--"
        return "IN" if self.within_profile_radius else "OUT"


@dataclass(frozen=True, slots=True)
class FeatureProfileDetail:
    device_id: str
    distance: float
    threshold: float
    normalized_distance: float
    tolerance: float
    is_selected: bool

    @property
    def range_status(self) -> str:
        return "IN" if self.normalized_distance <= self.tolerance else "OUT"


@dataclass(frozen=True, slots=True)
class AreaProfileDetail:
    device_id: str
    normalized_distance: float
    tolerance: float
    is_selected: bool

    @property
    def range_status(self) -> str:
        return "IN" if self.normalized_distance <= self.tolerance else "OUT"


@dataclass(frozen=True, slots=True)
class GazeSnapshot:
    """Everything the Gaze pipeline produced for one frame, for display.

    Fields map 1:1 to the pipeline stages so each panel renders one stage:
    landmarks → vector → smoothing → classifier → lock → contract.
    """

    timestamp_ms: int
    frame_id: int

    # 2a — landmarks (FaceObservation)
    face_detected: bool
    head_yaw_deg: float
    head_pitch_deg: float
    head_roll_deg: float
    left_iris_relative: tuple[float, float]
    right_iris_relative: tuple[float, float]
    left_eye_center_normalized: tuple[float, float] | None
    right_eye_center_normalized: tuple[float, float] | None
    face_scale: float | None
    tracking_confidence: float
    eyes_open: bool

    # 2b — composed gaze vector (None when tracking is lost / rejected)
    raw_gaze_direction: tuple[float, float, float] | None
    raw_gaze_confidence: float | None
    gaze_direction: tuple[float, float, float] | None
    gaze_confidence: float | None
    calibration_applied: bool
    calibration_features: tuple[float, ...] | None

    # 2c — temporal smoothing
    smoothed_stability: float | None
    smoothed_gaze_direction: tuple[float, float, float] | None
    smoothed_gaze_origin: tuple[float, float, float] | None
    buffer_fill: int
    buffer_capacity: int

    # 2d — target classification
    target: str
    target_label: str
    probability: float
    second_best_probability: float
    margin: float
    reject_reason: str | None
    device_details: tuple[DeviceGazeDetail, ...]
    raw_device_details: tuple[DeviceGazeDetail, ...]
    feature_sample: TargetFeatureSample | None
    gaze_motion_delta_deg: tuple[float, float] | None
    feature_details: tuple[FeatureProfileDetail, ...]
    area_details: tuple[AreaProfileDetail, ...]
    camera_pose_warning: str | None

    # 2e — gaze lock state machine
    lock_state: GazeLockState
    locked_device: str | None
    is_confident: bool

    # 2f — the contract message emitted to Fusion
    target_estimate: TargetEstimate

    # per-frame gaze compute time (capture→estimate), measured, ms
    inference_ms: float

    @property
    def tracking_lost(self) -> bool:
        return self.gaze_direction is None and self.smoothed_gaze_direction is None

    @property
    def tracking_recovering(self) -> bool:
        return not self.face_detected and self.smoothed_gaze_direction is not None


def _reject_reason(
    result: ClassificationResult,
    details: tuple[DeviceGazeDetail, ...],
    config: GazeConfig,
    feature_details: tuple[FeatureProfileDetail, ...] = (),
) -> str | None:
    """Explain why the classifier returned UNKNOWN, including profile range diagnostics."""
    if result.target != config.UNKNOWN_TARGET:
        return None
    if feature_details:
        nearest_feature = feature_details[0]
        if nearest_feature.normalized_distance > config.target_match_tolerance:
            return (
                f"nearest feature profile OUT: "
                f"{nearest_feature.distance:.2f}/{nearest_feature.threshold:.2f} "
                f"(x{nearest_feature.normalized_distance:.2f})"
            )
    if not details:
        return "등록된 target 프로파일 없음 (calibration required)"

    best_angle = min(d.angular_distance_deg for d in details)
    nearest = min(details, key=lambda d: d.angular_distance_deg)
    if best_angle > config.unknown_max_angle_deg:
        return (
            f"nearest target angle {best_angle:.1f}deg > "
            f"{config.unknown_max_angle_deg:.0f}deg"
        )
    if nearest.normalized_distance > config.target_match_tolerance:
        return (
            f"nearest target range OUT: "
            f"{nearest.angular_distance_deg:.1f}deg / {nearest.allowed_radius_deg:.1f}deg "
            f"(x{nearest.normalized_distance:.2f})"
        )
    if result.probability < config.unknown_probability_threshold:
        return (
            f"top-1 probability {result.probability:.0%} < "
            f"{config.unknown_probability_threshold:.0%}"
        )
    margin = result.probability - result.second_best_probability
    if margin < config.minimum_margin:
        return f"top-1/top-2 margin {margin:.2f} < {config.minimum_margin:.2f}"
    return "UNKNOWN"


def _is_confident(result: ClassificationResult, config: GazeConfig) -> bool:
    """Mirror of lock._is_confident (kept local to avoid importing a private)."""
    if result.target == config.UNKNOWN_TARGET:
        return False
    margin = result.probability - result.second_best_probability
    return result.probability >= config.minimum_probability and margin >= config.minimum_margin


def _face_scale(observation: FaceObservation) -> float | None:
    left_eye = observation.left_eye_center_normalized
    right_eye = observation.right_eye_center_normalized
    if left_eye is None or right_eye is None:
        return None
    scale = math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
    return scale if math.isfinite(scale) and scale > 0.0 else None


def _iris_offset(observation: FaceObservation) -> tuple[float, float] | None:
    if not observation.eyes_open:
        return None
    left_x, left_y = observation.left_iris_relative
    right_x, right_y = observation.right_iris_relative
    return ((left_x + right_x) * 0.5, (left_y + right_y) * 0.5)


def _unstable_iris_reason(
    observation: FaceObservation,
    config: GazeConfig,
    *,
    previous_iris_offset: tuple[float, float] | None = None,
    last_closed_eye_ms: int | None = None,
) -> str | None:
    offset = _iris_offset(observation)
    if offset is None:
        return None
    if last_closed_eye_ms is not None:
        elapsed = observation.timestamp_ms - last_closed_eye_ms
        if 0 <= elapsed <= config.blink_recovery_hold_ms:
            return "blink recovery hold"
    if max(abs(offset[0]), abs(offset[1])) > config.max_valid_eye_offset:
        return "iris offset out of range"
    if previous_iris_offset is not None:
        jump = math.hypot(offset[0] - previous_iris_offset[0], offset[1] - previous_iris_offset[1])
        if jump > config.iris_jump_threshold:
            return f"iris jump {jump:.2f}"
    return None


def _feature_sample(
    observation: FaceObservation,
    direction: tuple[float, float, float] | None,
    face_scale: float | None,
) -> TargetFeatureSample | None:
    if direction is None or face_scale is None:
        return None
    gaze_yaw, gaze_pitch = direction_to_yaw_pitch(np.asarray(direction, dtype=np.float64))
    try:
        return TargetFeatureSample(
            gaze_yaw=gaze_yaw,
            gaze_pitch=gaze_pitch,
            head_yaw=observation.head_yaw_deg,
            head_pitch=observation.head_pitch_deg,
            head_roll=observation.head_roll_deg,
            face_scale=face_scale,
        )
    except ValueError:
        return None


def _feature_details(
    sample: TargetFeatureSample | None,
    classifier: TargetClassifier,
    config: GazeConfig,
    selected_target: str,
) -> tuple[FeatureProfileDetail, ...]:
    if sample is None:
        return tuple()
    details = []
    for device_id, profile in classifier.feature_profiles.items():
        distance = profile.mahalanobis_distance(sample)
        normalized = distance / profile.threshold
        details.append(
            FeatureProfileDetail(
                device_id=device_id,
                distance=distance,
                threshold=profile.threshold,
                normalized_distance=normalized,
                tolerance=config.target_match_tolerance,
                is_selected=device_id == selected_target,
            )
        )
    return tuple(sorted(details, key=lambda item: item.normalized_distance))


def _area_details(
    sample: TargetFeatureSample | None,
    classifier: TargetClassifier,
    config: GazeConfig,
    selected_target: str,
) -> tuple[AreaProfileDetail, ...]:
    if sample is None:
        return tuple()
    profiles = classifier.profiles
    details = [
        AreaProfileDetail(
            device_id=device_id,
            normalized_distance=profile.normalized_distance(
                sample.gaze_yaw,
                sample.gaze_pitch,
                config.registration_max_area_radius_deg,
                _area_radius_scale(
                    profiles.get(device_id),
                    sample.face_scale,
                    config,
                ),
            ),
            tolerance=config.target_match_tolerance,
            is_selected=device_id == selected_target,
        )
        for device_id, profile in classifier.area_profiles.items()
    ]
    return tuple(sorted(details, key=lambda item: item.normalized_distance))


def _area_radius_scale(
    profile: DeviceGazeProfile | None,
    current_face_scale: float,
    config: GazeConfig,
) -> float:
    if profile is None or profile.reference_face_scale is None or current_face_scale <= 0.0:
        return 1.0
    ratio = current_face_scale / profile.reference_face_scale
    if not math.isfinite(ratio) or ratio <= 0.0:
        return 1.0
    flex = config.target_area_scale_flex
    return min(1.0 + flex, max(1.0 - flex, ratio))


def _device_details(
    direction: tuple[float, float, float] | None,
    origin: tuple[float, float, float] | None,
    classifier: TargetClassifier,
    config: GazeConfig,
    selected_target: str,
) -> tuple[DeviceGazeDetail, ...]:
    """Distance and registered-range diagnostics for each registered device."""
    profiles = classifier.profiles
    if not profiles or direction is None:
        return tuple(
            DeviceGazeDetail(
                device_id=device_id,
                angular_distance_deg=math.nan,
                allowed_radius_deg=math.nan,
                normalized_distance=math.nan,
                target_yaw_deg=math.nan,
                target_pitch_deg=math.nan,
                within_profile_radius=False,
                is_selected=False,
            )
            for device_id in profiles
        )

    gaze = np.array(direction, dtype=np.float64)
    ray_origin: Vector3 | None = np.array(origin, dtype=np.float64) if origin is not None else None
    geometries = classifier.geometries
    details: list[DeviceGazeDetail] = []
    for device_id, profile in profiles.items():
        geometry = geometries.get(device_id) if config.enable_3d_target_matching else None
        target_direction = profile.mean_direction
        if geometry is not None and ray_origin is not None:
            to_target = geometry.center_mm - ray_origin
            depth = float(np.linalg.norm(to_target))
            if depth > 1.0:
                target_direction = to_target / depth
        target_yaw_deg, target_pitch_deg = direction_to_yaw_pitch(target_direction)
        angular_distance, variance = effective_distance_and_variance(
            gaze, ray_origin, profile, geometry, config
        )
        angular_distance_deg = math.degrees(angular_distance)
        allowed_radius_deg = math.degrees(math.sqrt(max(variance, 0.0)))
        normalized_distance = (
            angular_distance_deg / allowed_radius_deg if allowed_radius_deg > 0.0 else math.inf
        )
        details.append(
            DeviceGazeDetail(
                device_id=device_id,
                angular_distance_deg=angular_distance_deg,
                allowed_radius_deg=allowed_radius_deg,
                normalized_distance=normalized_distance,
                target_yaw_deg=target_yaw_deg,
                target_pitch_deg=target_pitch_deg,
                within_profile_radius=normalized_distance <= config.target_match_tolerance,
                is_selected=(device_id == selected_target),
            )
        )
    details.sort(key=lambda d: d.angular_distance_deg)
    return tuple(details)


def _camera_pose_warning(
    classifier: TargetClassifier,
    current_face_scale: float | None,
    head_roll_deg: float,
) -> str | None:
    reference_scales = [
        profile.reference_face_scale
        for profile in classifier.profiles.values()
        if profile.reference_face_scale is not None and profile.reference_face_scale > 0.0
    ]
    warnings: list[str] = []
    if reference_scales and current_face_scale is not None and current_face_scale > 0.0:
        reference = float(np.median(np.asarray(reference_scales, dtype=np.float64)))
        ratio = current_face_scale / reference
        if math.isfinite(ratio) and (ratio < 0.70 or ratio > 1.30):
            warnings.append(f"face scale x{ratio:.2f}")
    if abs(head_roll_deg) > 30.0:
        warnings.append(f"head roll {head_roll_deg:+.0f}deg")
    if not warnings:
        return None
    return "camera/user pose changed: " + ", ".join(warnings) + " — re-register targets"


def evaluate(
    observation: FaceObservation,
    *,
    smoother: GazeSmoother,
    classifier: TargetClassifier,
    lock: GazeLockStateMachine,
    config: GazeConfig,
    inference_ms: float = 0.0,
    target_labels: dict[str, str] | None = None,
    calibration_model: GazeCalibrationModel | None = None,
    previous_iris_offset: tuple[float, float] | None = None,
    last_closed_eye_ms: int | None = None,
    previous_feature_sample: TargetFeatureSample | None = None,
) -> GazeSnapshot:
    """Run one observation through the full pipeline, capturing every stage.

    Mirrors ``GazeTargetingEngine.process`` exactly (same steps, same order) so
    the resulting ``TargetEstimate`` is identical to what the engine emits; the
    only addition is that the intermediate values are kept for display.
    """
    blink_hold = observation.face_detected and not observation.eyes_open
    unstable_iris = _unstable_iris_reason(
        observation,
        config,
        previous_iris_offset=previous_iris_offset,
        last_closed_eye_ms=last_closed_eye_ms,
    )
    jumpy_iris = unstable_iris is not None and unstable_iris.startswith("iris jump")
    hold_gaze = blink_hold or (unstable_iris is not None and not jumpy_iris)
    raw_gaze_vector = None if hold_gaze else compose_gaze_vector(observation, config)
    if raw_gaze_vector is not None and jumpy_iris:
        raw_gaze_vector = GazeVector(
            direction=raw_gaze_vector.direction,
            confidence=min(raw_gaze_vector.confidence, config.ema_min_alpha),
            timestamp_ms=raw_gaze_vector.timestamp_ms,
            frame_id=raw_gaze_vector.frame_id,
            origin=raw_gaze_vector.origin,
        )
    calibration_features = (
        observation_features(observation, raw_gaze_vector)
        if raw_gaze_vector is not None
        else None
    )
    gaze_vector = raw_gaze_vector
    calibration_applied = False
    if raw_gaze_vector is not None and calibration_model is not None:
        corrected = calibration_model.correct(observation, raw_gaze_vector)
        calibration_applied = calibration_model.fitted and not np.allclose(
            corrected.direction, raw_gaze_vector.direction
        )
        gaze_vector = corrected
    smoothed = (
            smoother.update(gaze_vector)
            if gaze_vector is not None
            else smoother.hold(observation.timestamp_ms, observation.frame_id)
            if hold_gaze
            else smoother.hold_tracking_loss(observation.timestamp_ms, observation.frame_id)
        )

    raw_gaze_direction: tuple[float, float, float] | None = None
    raw_gaze_confidence: float | None = None
    gaze_direction: tuple[float, float, float] | None = None
    gaze_confidence: float | None = None
    if raw_gaze_vector is not None:
        raw_gaze_direction = (
            float(raw_gaze_vector.direction[0]),
            float(raw_gaze_vector.direction[1]),
            float(raw_gaze_vector.direction[2]),
        )
        raw_gaze_confidence = raw_gaze_vector.confidence
    if gaze_vector is not None:
        gaze_direction = (
            float(gaze_vector.direction[0]),
            float(gaze_vector.direction[1]),
            float(gaze_vector.direction[2]),
        )
        gaze_confidence = gaze_vector.confidence

    smoothed_stability: float | None = None
    classify_direction: tuple[float, float, float] | None = None
    classify_origin: tuple[float, float, float] | None = None
    current_face_scale = _face_scale(observation)
    feature_sample: TargetFeatureSample | None = None
    gaze_motion_delta_deg: tuple[float, float] | None = None
    if smoothed is None:
        result = ClassificationResult(
            target=config.UNKNOWN_TARGET, probability=0.0, second_best_probability=0.0
        )
        lock.update(observation.timestamp_ms, result)
        estimate = TargetEstimate(
            timestamp_ms=observation.timestamp_ms,
            frame_id=observation.frame_id,
            target=result.target,
            probability=result.probability,
            second_best_probability=result.second_best_probability,
            stability=0.0,
        )
    else:
        smoothed_stability = smoothed.stability
        classify_direction = (
            float(smoothed.direction[0]),
            float(smoothed.direction[1]),
            float(smoothed.direction[2]),
        )
        if smoothed.origin is not None:
            classify_origin = (
                float(smoothed.origin[0]),
                float(smoothed.origin[1]),
                float(smoothed.origin[2]),
            )
        feature_sample = _feature_sample(observation, classify_direction, current_face_scale)
        if feature_sample is not None and previous_feature_sample is not None:
            gaze_motion_delta_deg = (
                feature_sample.gaze_yaw - previous_feature_sample.gaze_yaw,
                feature_sample.gaze_pitch - previous_feature_sample.gaze_pitch,
            )
        result = classifier.classify(
            smoothed.direction,
            origin=smoothed.origin,
            current_face_scale=current_face_scale,
            feature_sample=feature_sample,
            gaze_motion_delta_deg=gaze_motion_delta_deg,
        )
        lock.update(smoothed.timestamp_ms, result)
        estimate = TargetEstimate(
            timestamp_ms=smoothed.timestamp_ms,
            frame_id=smoothed.frame_id,
            target=result.target,
            probability=result.probability,
            second_best_probability=result.second_best_probability,
            stability=smoothed.stability,
        )

    details = _device_details(classify_direction, classify_origin, classifier, config, result.target)
    raw_details = _device_details(
        raw_gaze_direction,
        classify_origin,
        classifier,
        config,
        result.target,
    )
    profile_details = _feature_details(feature_sample, classifier, config, result.target)
    area_details = _area_details(feature_sample, classifier, config, result.target)
    target_label = (
        config.UNKNOWN_TARGET
        if result.target == config.UNKNOWN_TARGET
        else (target_labels or {}).get(result.target, result.target)
    )

    return GazeSnapshot(
        timestamp_ms=observation.timestamp_ms,
        frame_id=observation.frame_id,
        face_detected=observation.face_detected,
        head_yaw_deg=observation.head_yaw_deg,
        head_pitch_deg=observation.head_pitch_deg,
        head_roll_deg=observation.head_roll_deg,
        left_iris_relative=observation.left_iris_relative,
        right_iris_relative=observation.right_iris_relative,
        left_eye_center_normalized=observation.left_eye_center_normalized,
        right_eye_center_normalized=observation.right_eye_center_normalized,
        face_scale=current_face_scale,
        tracking_confidence=min(
            observation.eye_tracking_confidence, observation.face_tracking_confidence
        ),
        eyes_open=observation.eyes_open,
        raw_gaze_direction=raw_gaze_direction,
        raw_gaze_confidence=raw_gaze_confidence,
        gaze_direction=gaze_direction,
        gaze_confidence=gaze_confidence,
        calibration_applied=calibration_applied,
        calibration_features=calibration_features,
        smoothed_stability=smoothed_stability,
        smoothed_gaze_direction=classify_direction,
        smoothed_gaze_origin=classify_origin,
        buffer_fill=len(smoother._buffer),  # noqa: SLF001 - diagnostic read of buffer depth
        buffer_capacity=config.smoothing_window_frames,
        target=result.target,
        target_label=target_label,
        probability=result.probability,
        second_best_probability=result.second_best_probability,
        margin=result.probability - result.second_best_probability,
        reject_reason=_reject_reason(result, details, config, profile_details),
        device_details=details,
        raw_device_details=raw_details,
        feature_sample=feature_sample,
        gaze_motion_delta_deg=gaze_motion_delta_deg,
        feature_details=profile_details,
        area_details=area_details,
        camera_pose_warning=_camera_pose_warning(
            classifier,
            current_face_scale,
            observation.head_roll_deg,
        ),
        lock_state=lock.state,
        locked_device=lock.locked_device,
        is_confident=_is_confident(result, config),
        target_estimate=estimate,
        inference_ms=inference_ms,
    )


class GazeProbe:
    """Owns the live gaze pipeline state and turns BGR frames into snapshots.

    Holds one smoother/classifier/lock (stateful across frames, exactly like the
    engine). The MediaPipe adapter is created lazily so this class can be
    constructed and its liveness checked without the ``vision`` extra.
    """

    def __init__(
        self,
        *,
        model_path: Path | None,
        profiles_path: Path | None = None,
        config: GazeConfig | None = None,
        calibration_model: GazeCalibrationModel | None = None,
    ) -> None:
        self._config = config or GazeConfig()
        self._smoother = GazeSmoother(self._config)
        self._classifier = TargetClassifier(self._config)
        self._lock = GazeLockStateMachine(self._config)
        self._model_path = model_path
        self._adapter: object | None = None
        self._available = False
        self._status_text = "gaze 프로브 미시작"
        self._target_labels: dict[str, str] = {}
        self._calibration_model = calibration_model
        self._previous_iris_offset: tuple[float, float] | None = None
        self._last_closed_eye_ms: int | None = None
        self._previous_feature_sample: TargetFeatureSample | None = None
        self._profile_count = self._load_profiles(profiles_path)

    def _load_profiles(self, profiles_path: Path | None) -> int:
        """Load every registered target, including 3D geometry when present.

        Uses `TargetRegistry` (not the flat `calibration.profiles` loader) so
        `position_3d` survives an app restart — loading only the flat angular
        profile here would silently drop 3D geometry every relaunch even
        though the JSON file still has it (`TargetRegistry` is the same file
        `app.py` writes to via `self._profiles_path`).
        """
        if profiles_path is None or not profiles_path.is_file():
            return 0
        from jarvis.calibration.registry import TargetRegistry

        registry = TargetRegistry(profiles_path)
        count = 0
        for record in registry.records:
            self._classifier.register_profile(
                record.to_profile(),
                geometry_3d=record.to_geometry_3d(),
                feature_profile=record.feature_profile,
                area_profile=record.area_profile,
            )
            self._target_labels[record.target_id] = record.name
            count += 1
        return count

    @property
    def available(self) -> bool:
        return self._available

    @property
    def status_text(self) -> str:
        return self._status_text

    @property
    def profile_count(self) -> int:
        return self._profile_count

    def set_calibration_model(self, model: GazeCalibrationModel | None) -> None:
        self._calibration_model = model

    def register_profile(
        self,
        profile: DeviceGazeProfile,
        geometry_3d: TargetGeometry3D | None = None,
        feature_profile: TargetFeatureProfile | None = None,
        area_profile: TargetAreaProfile | None = None,
        label: str | None = None,
    ) -> None:
        """Add or replace one target profile in the live classifier."""
        existed = profile.device_id in self._classifier.profiles
        self._classifier.register_profile(
            profile,
            geometry_3d=geometry_3d,
            feature_profile=feature_profile,
            area_profile=area_profile,
        )
        if label is not None:
            self._target_labels[profile.device_id] = label
        if not existed:
            self._profile_count += 1

    def unregister_profile(self, device_id: str) -> None:
        """Remove one target profile from the live classifier."""
        existed = device_id in self._classifier.profiles
        self._classifier.unregister_profile(device_id)
        self._target_labels.pop(device_id, None)
        if existed:
            self._profile_count = max(0, self._profile_count - 1)

    def start(self) -> bool:
        """Create the MediaPipe adapter. Returns True on success.

        Failure (mediapipe missing, model file absent) sets an honest
        ``status_text`` and leaves the probe unavailable — it never pretends.
        """
        if self._model_path is None or not self._model_path.is_file():
            self._status_text = "face_landmarker.task 모델 없음 (models/README.md 참고)"
            return False
        try:
            from jarvis.gaze.landmarks import FaceLandmarkerAdapter
        except ImportError:
            self._status_text = "mediapipe 미설치 — pip install -e \".[vision]\""
            return False
        try:
            self._adapter = FaceLandmarkerAdapter(self._model_path)
        except Exception as exc:  # noqa: BLE001 - surface any init failure honestly
            self._status_text = f"gaze 어댑터 초기화 실패: {exc}"
            return False
        self._available = True
        model_name = self._model_path.name
        profiles = f"{self._profile_count}개 기기 등록됨" if self._profile_count else "프로파일 없음"
        self._status_text = f"LIVE · {model_name} · {profiles}"
        return True

    def process_bgr(
        self, bgr_frame: npt.NDArray[np.uint8], timestamp_ms: int, frame_id: int
    ) -> GazeSnapshot | None:
        """Convert a BGR frame, run landmarks + pipeline, return a snapshot.

        Returns ``None`` when the probe is not available (no adapter). Timing is
        measured across landmark detection + the full pipeline (capture→estimate).
        """
        if self._adapter is None:
            return None
        import cv2

        rgb = cast("npt.NDArray[np.uint8]", cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        return self.process_rgb(rgb, timestamp_ms, frame_id)

    def process_rgb(
        self, rgb_frame: npt.NDArray[np.uint8], timestamp_ms: int, frame_id: int
    ) -> GazeSnapshot | None:
        """Same as :meth:`process_bgr` for an already-converted RGB frame.

        Lets a caller that feeds several probes (camera worker) convert the
        frame once instead of once per probe.
        """
        if self._adapter is None:
            return None
        import time

        from jarvis.gaze.landmarks import FaceLandmarkerAdapter

        assert isinstance(self._adapter, FaceLandmarkerAdapter)
        started = time.monotonic()
        observation = self._adapter.process(rgb_frame, timestamp_ms, frame_id)
        snapshot = evaluate(
            observation,
            smoother=self._smoother,
            classifier=self._classifier,
            lock=self._lock,
            config=self._config,
            inference_ms=(time.monotonic() - started) * 1000.0,
            target_labels=self._target_labels,
            calibration_model=self._calibration_model,
            previous_iris_offset=self._previous_iris_offset,
            last_closed_eye_ms=self._last_closed_eye_ms,
            previous_feature_sample=self._previous_feature_sample,
        )
        if observation.face_detected and not observation.eyes_open:
            self._last_closed_eye_ms = observation.timestamp_ms
        if snapshot.raw_gaze_direction is not None:
            self._previous_iris_offset = _iris_offset(observation)
        if snapshot.feature_sample is not None:
            self._previous_feature_sample = snapshot.feature_sample
        return snapshot

    def close(self) -> None:
        if self._adapter is not None:
            from jarvis.gaze.landmarks import FaceLandmarkerAdapter

            assert isinstance(self._adapter, FaceLandmarkerAdapter)
            self._adapter.close()
            self._adapter = None
            self._available = False
