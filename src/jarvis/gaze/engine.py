"""GazeTargetingEngine: per-frame orchestration producing `TargetEstimate`.

    FaceObservation
    → compose_gaze_vector (features.py)
    → GazeSmoother (smoothing.py)
    → TargetClassifier (classifier.py)
    → GazeLockStateMachine (lock.py)
    → jarvis.contracts.TargetEstimate (documents/interface-contract.md 1번 계약)

`src/jarvis/gaze/README.md`가 정한 대로, 이 엔진이 외부로 내보내는 값은
`jarvis.contracts.TargetEstimate` 하나뿐이다.
"""

from __future__ import annotations

from jarvis.contracts.messages import TargetEstimate
from jarvis.gaze.classifier import (
    ClassificationResult,
    DeviceGazeProfile,
    TargetClassifier,
    TargetGeometry3D,
)
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.features import FaceObservation, compose_gaze_vector, compose_head_vector
from jarvis.gaze.lock import GazeLockState, GazeLockStateMachine
from jarvis.gaze.nod import NodDetector
from jarvis.gaze.smoothing import GazeSmoother, SmoothedGaze


class GazeTargetingEngine:
    """Gaze Targeting Engine의 조립 지점(composition root 바로 아래 계층)."""

    def __init__(self, config: GazeConfig = GazeConfig()) -> None:
        self._config = config
        self._smoother = GazeSmoother(config)
        self._classifier = TargetClassifier(config)
        self._lock = GazeLockStateMachine(config)
        self._nod_detector = NodDetector(config)
        self._nod_gated_devices: set[str] = set()
        self._last_smoothed_gaze: SmoothedGaze | None = None

    @property
    def lock_state(self) -> GazeLockState:
        return self._lock.state

    @property
    def last_smoothed_gaze(self) -> SmoothedGaze | None:
        """가장 최근 classifier 입력과 동일한 평활화 시선 벡터."""
        return self._last_smoothed_gaze

    def is_gaze_locked_to(self, device_id: str) -> bool:
        """Cursor Control Mapper 게이트(README 6장 `Gaze Lock == laptop`)에서 쓴다."""
        return self._lock.is_locked_to(device_id)

    def register_device(
        self,
        profile: DeviceGazeProfile,
        geometry_3d: TargetGeometry3D | None = None,
        *,
        requires_nod_gate: bool = False,
    ) -> None:
        """`requires_nod_gate=True`면 다른 target이 확정된 뒤 이 target으로
        돌아올 때만 끄덕임 확인을 요구한다(lock.py의 게이트, 2026-07-22:
        카메라 정면 방향에 놓인 target이 "가만히 있을 때의 기본 시선 방향"과
        겹쳐 생기는 오확정을 막기 위함). 같은 target을 계속 보던 중이거나
        세션 최초 확정에는 적용되지 않는다."""
        self._classifier.register_profile(profile, geometry_3d=geometry_3d)
        if requires_nod_gate:
            self._nod_gated_devices.add(profile.device_id)
        else:
            self._nod_gated_devices.discard(profile.device_id)

    def unregister_device(self, device_id: str) -> None:
        self._classifier.unregister_profile(device_id)
        self._nod_gated_devices.discard(device_id)

    def notify_gesture_started(self, timestamp_ms: int) -> GazeLockState:
        """Fusion이 Target Lock 상태에서 gesture 시작을 감지했을 때 호출한다."""
        return self._lock.notify_gesture_started(timestamp_ms)

    def notify_committed(self, timestamp_ms: int) -> GazeLockState:
        """Fusion이 GESTURE_WAIT 상태에서 intent를 commit했을 때 호출한다."""
        return self._lock.notify_committed(timestamp_ms)

    def process(self, observation: FaceObservation) -> TargetEstimate:
        """한 프레임을 처리해 Gaze→Fusion 계약(TargetEstimate)을 만든다.

        추적 손실이나 등록된 기기가 없을 때도 항상 유효한 TargetEstimate를
        반환한다 — 이때 target은 `config.UNKNOWN_TARGET`이고 probability·
        stability는 0.0이다(성공을 지어내지 않는다, development-principles.md 1절).
        """
        nod_detected = (
            self._nod_detector.update(observation.head_pitch_deg, observation.timestamp_ms)
            if observation.face_detected
            else False
        )
        blink_hold = observation.face_detected and not observation.eyes_open
        gaze_vector = None if blink_hold else compose_gaze_vector(observation, self._config)
        if gaze_vector is not None:
            smoothed = self._smoother.update(gaze_vector)
        elif blink_hold:
            if self._smoother.last_source == "head-only":
                fallback = compose_head_vector(observation, self._config)
                smoothed = self._smoother.update(fallback) if fallback is not None else None
            else:
                smoothed = self._smoother.hold(observation.timestamp_ms, observation.frame_id)
                if smoothed is None:
                    fallback = compose_head_vector(observation, self._config)
                    smoothed = self._smoother.update(fallback) if fallback is not None else None
        else:
            smoothed = self._smoother.hold_tracking_loss(
                observation.timestamp_ms, observation.frame_id
            )
        self._last_smoothed_gaze = smoothed

        if smoothed is None:
            classification = ClassificationResult(
                target=self._config.UNKNOWN_TARGET,
                probability=0.0,
                second_best_probability=0.0,
            )
            self._lock.update(
                observation.timestamp_ms,
                classification,
                candidate_requires_nod_gate=classification.target in self._nod_gated_devices,
                nod_detected=nod_detected,
            )
            return TargetEstimate(
                timestamp_ms=observation.timestamp_ms,
                frame_id=observation.frame_id,
                target=classification.target,
                probability=classification.probability,
                second_best_probability=classification.second_best_probability,
                stability=0.0,
            )

        classification = self._classifier.classify(smoothed.direction, origin=smoothed.origin)
        self._lock.update(
            smoothed.timestamp_ms,
            classification,
            candidate_requires_nod_gate=classification.target in self._nod_gated_devices,
            nod_detected=nod_detected,
        )

        return TargetEstimate(
            timestamp_ms=smoothed.timestamp_ms,
            frame_id=smoothed.frame_id,
            target=classification.target,
            probability=classification.probability,
            second_best_probability=classification.second_best_probability,
            stability=smoothed.stability,
        )
