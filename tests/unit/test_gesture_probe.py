"""GestureProbe·ProbeGestureSource 검증 — mediapipe·카메라 없이 파이프라인 조립을 테스트한다.

`process_bgr`(mediapipe 필요)은 여기서 다루지 않고, 그 뒤 순수 파이프라인
(`_advance`: feature→model→spotting)과 사이드바 소스 버퍼링을 검증한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.contracts.messages import GesturePhase
from jarvis.gesture_fusion import DEFAULT_GESTURE_CONFIG, feature_dimension
from jarvis.gesture_fusion.config import HAND_LANDMARK_COUNT, LANDMARK_DIMS
from jarvis.gesture_fusion.landmarks import HandObservation
from jarvis.gesture_fusion.model_protocol import (
    DEFAULT_GESTURE_LABELS,
    ModelPrediction,
    SlidingFeatureWindow,
)
from jarvis.monitoring.gesture_probe import GestureProbe, GestureSnapshot, ProbeGestureSource


class _FakeModel:
    """고정 예측을 내는 GestureModel — 파이프라인 조립만 검증."""

    def __init__(self, phase: GesturePhase, gesture: str = "swipe_down") -> None:
        self._phase = phase
        self._gesture = gesture

    @property
    def labels(self) -> tuple[str, ...]:
        return DEFAULT_GESTURE_LABELS

    @property
    def window_size(self) -> int:
        return 4

    def predict(self, window: np.ndarray) -> ModelPrediction:
        return ModelPrediction(
            gesture=self._gesture, gesture_confidence=0.9,
            phase=self._phase, phase_confidence=0.9, uncertainty=0.05,
        )


def _observation(*, hand_detected: bool, timestamp_ms: int, frame_id: int) -> HandObservation:
    if not hand_detected:
        return HandObservation(
            timestamp_ms=timestamp_ms, frame_id=frame_id,
            landmarks=np.zeros((HAND_LANDMARK_COUNT, LANDMARK_DIMS)), handedness="",
            palm_scale=0.0, detection_confidence=0.0, handedness_score=0.0, hand_detected=False,
        )
    pts = np.zeros((HAND_LANDMARK_COUNT, LANDMARK_DIMS), dtype=np.float64)
    for j in range(HAND_LANDMARK_COUNT):
        pts[j] = [j * 0.1, 0.0]
    return HandObservation(
        timestamp_ms=timestamp_ms, frame_id=frame_id, landmarks=pts, handedness="Right",
        palm_scale=0.2, detection_confidence=0.9, handedness_score=0.95, hand_detected=True,
    )


def _ready_probe(model: object) -> GestureProbe:
    """start() 없이(mediapipe 불필요) 파이프라인만 돌릴 수 있게 모델·윈도우를 채운다."""
    probe = GestureProbe(model_asset_path=None, model=model)  # type: ignore[arg-type]
    probe._model = model  # type: ignore[assignment]
    probe._window = SlidingFeatureWindow(
        window_size=model.window_size,  # type: ignore[attr-defined]
        feature_dim=feature_dimension(DEFAULT_GESTURE_CONFIG),
    )
    return probe


def test_unavailable_without_model_file() -> None:
    probe = GestureProbe(model_asset_path=None)
    assert probe.start() is False
    assert probe.available is False
    assert "hand_landmarker.task" in probe.status_text


def test_advance_produces_estimate_from_detected_hand() -> None:
    probe = _ready_probe(_FakeModel(GesturePhase.ONSET))
    import time
    for i in range(3):  # 디바운스 통과하도록 여러 프레임
        snap = probe._advance(_observation(hand_detected=True, timestamp_ms=i * 33, frame_id=i), time.monotonic())
    assert isinstance(snap, GestureSnapshot)
    assert snap.hand_detected
    assert snap.estimate.gesture == "swipe_down"
    assert snap.landmarks.shape == (HAND_LANDMARK_COUNT, LANDMARK_DIMS)


def test_lost_tracking_yields_idle_and_resets() -> None:
    import time
    probe = _ready_probe(_FakeModel(GesturePhase.ACTIVE))
    snap = probe._advance(_observation(hand_detected=False, timestamp_ms=0, frame_id=0), time.monotonic())
    assert not snap.hand_detected
    assert snap.estimate.phase == GesturePhase.IDLE


def test_advance_runs_real_untrained_torch_model() -> None:
    """실제 Causal TCN(미학습)을 프로브 파이프라인에 통과시켜 실행됨을 검증."""
    pytest.importorskip("torch")
    from jarvis.gesture_fusion.model import CausalTCNGestureModel, ModelConfig

    model = CausalTCNGestureModel(ModelConfig(feature_dim=feature_dimension(DEFAULT_GESTURE_CONFIG)))
    probe = _ready_probe(model)
    import time
    snap = None
    for i in range(6):
        snap = probe._advance(_observation(hand_detected=True, timestamp_ms=i * 33, frame_id=i), time.monotonic())
    assert snap is not None and snap.hand_detected
    assert snap.estimate.gesture in DEFAULT_GESTURE_LABELS  # 미학습이라 값은 랜덤이지만 label 집합 안
    assert 0.0 <= snap.estimate.gesture_confidence <= 1.0


def test_probe_source_buffers_only_phase_transitions() -> None:
    source = ProbeGestureSource(_ready_probe(_FakeModel(GesturePhase.ONSET)))

    def _snap(phase: GesturePhase, ts: int) -> GestureSnapshot:
        from jarvis.contracts.messages import GestureEstimate
        est = GestureEstimate(
            timestamp_ms=ts, frame_id=ts, gesture="swipe_down",
            gesture_confidence=0.9, phase=phase, phase_confidence=0.9, uncertainty=0.05,
        )
        return GestureSnapshot(
            timestamp_ms=ts, frame_id=ts, hand_detected=True, estimate=est,
            landmarks=np.zeros((HAND_LANDMARK_COUNT, LANDMARK_DIMS)), latency_ms=1.0,
        )

    source.submit(_snap(GesturePhase.IDLE, 0))     # IDLE은 안 쌓음
    source.submit(_snap(GesturePhase.ONSET, 33))   # 전이 → 쌓음
    source.submit(_snap(GesturePhase.ONSET, 66))   # 같은 phase → 안 쌓음
    source.submit(_snap(GesturePhase.ACTIVE, 99))  # 전이 → 쌓음
    source.submit(None)                            # None 무시

    drained = source.poll()
    assert [g.phase for g in drained] == ["ONSET", "ACTIVE"]
    assert source.poll() == []  # 비워짐


def test_probe_source_reports_probe_availability() -> None:
    source = ProbeGestureSource(GestureProbe(model_asset_path=None))
    assert source.available is False
    assert isinstance(source.status_text, str)
