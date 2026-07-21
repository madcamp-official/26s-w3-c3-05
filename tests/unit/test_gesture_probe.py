"""GestureProbe·ProbeGestureSource 검증 — mediapipe·카메라 없이 파이프라인 조립을 테스트한다.

`process_bgr`(mediapipe 필요)은 여기서 다루지 않고, 그 뒤 순수 파이프라인
(`_advance`: feature→model→spotting)과 사이드바 소스 버퍼링을 검증한다.
"""

from __future__ import annotations

from pathlib import Path

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


def test_probe_defaults_to_expected_input_fps_rate_limiter() -> None:
    """추론은 기본적으로 학습 cadence(EXPECTED_INPUT_FPS)로 프레임을 솎아야 한다."""
    from jarvis.gesture_fusion.model_protocol import EXPECTED_INPUT_FPS

    probe = GestureProbe(model_asset_path=None)
    assert probe._rate_limiter is not None
    assert probe._rate_limiter.target_fps == EXPECTED_INPUT_FPS


def test_probe_rate_limiter_can_be_disabled() -> None:
    """target_fps=None이면 솎지 않는다(모든 프레임 처리)."""
    probe = GestureProbe(model_asset_path=None, target_fps=None)
    assert probe._rate_limiter is None


# --- 관측값 재사용(headless) 모드: HandProbe 관측값으로 인식 구동 ---


def test_activate_headless_prepares_pipeline_without_landmarker() -> None:
    probe = GestureProbe(model_asset_path=None, model=_FakeModel(GesturePhase.ONSET))
    assert probe.activate_headless() is True
    assert probe.available is True
    assert probe._landmarker is None  # 두 번째 landmarker를 돌리지 않는다


def test_activate_headless_false_without_model() -> None:
    probe = GestureProbe(model_asset_path=None)
    assert probe.activate_headless() is False


def test_advance_from_observation_produces_estimate() -> None:
    probe = GestureProbe(model_asset_path=None, model=_FakeModel(GesturePhase.ONSET))
    probe.activate_headless()
    snap = None
    for ts in (0, 100, 200):  # 100ms 간격 = 전부 채택(12fps=83ms 초과), 디바운스 통과
        snap = probe.advance(_observation(hand_detected=True, timestamp_ms=ts, frame_id=ts))
    assert snap is not None and snap.hand_detected
    assert snap.estimate.gesture == "swipe_down"


def test_advance_decimates_to_target_fps() -> None:
    """30fps(33ms) 입력에서 target 간격 미만 프레임은 skip돼 직전 스냅샷을 돌려준다."""
    probe = GestureProbe(model_asset_path=None, model=_FakeModel(GesturePhase.ONSET))
    probe.activate_headless()
    snap0 = probe.advance(_observation(hand_detected=True, timestamp_ms=0, frame_id=0))
    snap1 = probe.advance(_observation(hand_detected=True, timestamp_ms=33, frame_id=1))
    assert snap1 is snap0  # 33ms < 83ms → skip, 같은 스냅샷 재사용
    snap2 = probe.advance(_observation(hand_detected=True, timestamp_ms=100, frame_id=3))
    assert snap2 is not snap0  # 100ms ≥ 83ms → 새로 처리


def test_advance_lost_frame_resets_and_is_always_processed() -> None:
    """손실 프레임은 솎지 않고 항상 처리해 파이프라인을 즉시 리셋한다."""
    probe = GestureProbe(model_asset_path=None, model=_FakeModel(GesturePhase.ACTIVE))
    probe.activate_headless()
    probe.advance(_observation(hand_detected=True, timestamp_ms=0, frame_id=0))
    lost = probe.advance(_observation(hand_detected=False, timestamp_ms=10, frame_id=1))
    assert lost is not None and not lost.hand_detected  # 10ms여도 skip되지 않음
    assert lost.estimate.phase == GesturePhase.IDLE


def test_load_trained_gesture_model_returns_none_without_checkpoint(tmp_path: Path) -> None:
    from jarvis.monitoring.gesture_probe import load_trained_gesture_model

    assert load_trained_gesture_model(tmp_path) is None  # 빈 디렉토리


def test_load_trained_gesture_model_returns_none_without_sidecar(tmp_path: Path) -> None:
    from jarvis.monitoring.gesture_probe import load_trained_gesture_model

    (tmp_path / "gesture_tcn_jester.pt").write_bytes(b"not-a-real-checkpoint")
    # 사이드카 metadata가 없으면 로드 대상으로 인정하지 않는다
    assert load_trained_gesture_model(tmp_path) is None


def test_load_trained_gesture_model_loads_trained_checkpoint(tmp_path: Path) -> None:
    """torch가 있으면 학습된(trained=True) 체크포인트를 로드하고, 미학습은 거부한다."""
    pytest.importorskip("torch")
    import json

    import torch

    from jarvis.gesture_fusion.model import CausalTCN, ModelConfig
    from jarvis.monitoring.gesture_probe import load_trained_gesture_model

    net = CausalTCN(ModelConfig(feature_dim=feature_dimension(DEFAULT_GESTURE_CONFIG)))
    torch.save(net.state_dict(), tmp_path / "gesture_tcn_jester.pt")

    # trained=False → 미학습이므로 거부(무작위 가중치를 인식 결과로 내보내지 않는다)
    (tmp_path / "gesture_tcn_jester.pt.metadata.json").write_text(
        json.dumps({"version": "x", "trained": False}), encoding="utf-8"
    )
    assert load_trained_gesture_model(tmp_path) is None

    # trained=True → 로드
    (tmp_path / "gesture_tcn_jester.pt.metadata.json").write_text(
        json.dumps({"version": "pretrain-epoch24", "trained": True}), encoding="utf-8"
    )
    model = load_trained_gesture_model(tmp_path)
    assert model is not None
    assert model.metadata.trained is True  # type: ignore[attr-defined]


def test_load_trained_gesture_model_prefers_finetuned(tmp_path: Path) -> None:
    """finetuned·jester가 둘 다 있으면 finetuned를 우선한다."""
    pytest.importorskip("torch")
    import json

    import torch

    from jarvis.gesture_fusion.model import CausalTCN, ModelConfig
    from jarvis.monitoring.gesture_probe import load_trained_gesture_model

    net = CausalTCN(ModelConfig(feature_dim=feature_dimension(DEFAULT_GESTURE_CONFIG)))
    for name, version in (
        ("gesture_tcn_finetuned.pt", "finetune-v1"),
        ("gesture_tcn_jester.pt", "pretrain-epoch24"),
    ):
        torch.save(net.state_dict(), tmp_path / name)
        (tmp_path / f"{name}.metadata.json").write_text(
            json.dumps({"version": version, "trained": True}), encoding="utf-8"
        )
    model = load_trained_gesture_model(tmp_path)
    assert model is not None
    assert model.metadata.version == "finetune-v1"  # type: ignore[attr-defined]
