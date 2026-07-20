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
from jarvis.gaze.classifier import (
    ClassificationResult,
    DeviceGazeProfile,
    TargetClassifier,
    TargetGeometry3D,
    effective_distance_and_variance,
)
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.features import FaceObservation, Vector3, compose_gaze_vector
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
    within_profile_radius: bool
    is_selected: bool

    @property
    def range_status(self) -> str:
        """Human-readable live range check for the registered target profile."""
        if math.isnan(self.angular_distance_deg):
            return "--"
        return "IN" if self.within_profile_radius else "OUT"


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
    tracking_confidence: float
    eyes_open: bool

    # 2b — composed gaze vector (None when tracking is lost / rejected)
    gaze_direction: tuple[float, float, float] | None
    gaze_confidence: float | None

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
        return not self.face_detected or self.gaze_direction is None


def _reject_reason(
    result: ClassificationResult,
    details: tuple[DeviceGazeDetail, ...],
    config: GazeConfig,
) -> str | None:
    """Explain why the classifier returned UNKNOWN, including profile range diagnostics."""
    if result.target != config.UNKNOWN_TARGET:
        return None
    if not details:
        return "등록된 target 프로파일 없음 (calibration required)"

    best_angle = min(d.angular_distance_deg for d in details)
    nearest = min(details, key=lambda d: d.angular_distance_deg)
    if best_angle > config.unknown_max_angle_deg:
        return (
            f"nearest target angle {best_angle:.1f}deg > "
            f"{config.unknown_max_angle_deg:.0f}deg"
        )
    if nearest.normalized_distance > 1.0:
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
                within_profile_radius=normalized_distance <= 1.0,
                is_selected=(device_id == selected_target),
            )
        )
    details.sort(key=lambda d: d.angular_distance_deg)
    return tuple(details)


def evaluate(
    observation: FaceObservation,
    *,
    smoother: GazeSmoother,
    classifier: TargetClassifier,
    lock: GazeLockStateMachine,
    config: GazeConfig,
    inference_ms: float = 0.0,
    target_labels: dict[str, str] | None = None,
) -> GazeSnapshot:
    """Run one observation through the full pipeline, capturing every stage.

    Mirrors ``GazeTargetingEngine.process`` exactly (same steps, same order) so
    the resulting ``TargetEstimate`` is identical to what the engine emits; the
    only addition is that the intermediate values are kept for display.
    """
    gaze_vector = compose_gaze_vector(observation, config)
    smoothed = (
        smoother.hold(observation.timestamp_ms, observation.frame_id)
        if observation.face_detected and not observation.eyes_open
        else smoother.update(gaze_vector)
    )

    gaze_direction: tuple[float, float, float] | None = None
    gaze_confidence: float | None = None
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
        result = classifier.classify(smoothed.direction, origin=smoothed.origin)
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
        tracking_confidence=min(
            observation.eye_tracking_confidence, observation.face_tracking_confidence
        ),
        eyes_open=observation.eyes_open,
        gaze_direction=gaze_direction,
        gaze_confidence=gaze_confidence,
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
        reject_reason=_reject_reason(result, details, config),
        device_details=details,
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
            self._classifier.register_profile(record.to_profile(), geometry_3d=record.to_geometry_3d())
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

    def register_profile(
        self,
        profile: DeviceGazeProfile,
        geometry_3d: TargetGeometry3D | None = None,
        label: str | None = None,
    ) -> None:
        """Add or replace one target profile in the live classifier."""
        existed = profile.device_id in self._classifier.profiles
        self._classifier.register_profile(profile, geometry_3d=geometry_3d)
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
        import time

        import cv2

        from jarvis.gaze.landmarks import FaceLandmarkerAdapter

        assert isinstance(self._adapter, FaceLandmarkerAdapter)
        started = time.monotonic()
        rgb = cast("npt.NDArray[np.uint8]", cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        observation = self._adapter.process(rgb, timestamp_ms, frame_id)
        snapshot = evaluate(
            observation,
            smoother=self._smoother,
            classifier=self._classifier,
            lock=self._lock,
            config=self._config,
            inference_ms=(time.monotonic() - started) * 1000.0,
            target_labels=self._target_labels,
        )
        return snapshot

    def close(self) -> None:
        if self._adapter is not None:
            from jarvis.gaze.landmarks import FaceLandmarkerAdapter

            assert isinstance(self._adapter, FaceLandmarkerAdapter)
            self._adapter.close()
            self._adapter = None
            self._available = False
