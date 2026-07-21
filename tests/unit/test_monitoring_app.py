"""Offscreen smoke test: the desktop window constructs and wires its panels.

Runs Qt under the 'offscreen' platform so it needs no display. This does not
verify visual appearance — only that the widget tree builds, the tabs exist, and
teardown is clean. Requires the ui extra (PySide6).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from jarvis.monitoring.app import MainWindow, _REGISTRATION_GUIDANCE_PHASES  # noqa: E402


def test_main_window_builds_offscreen(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        env={},
        start_camera=False,
        samples_path=tmp_path / "samples.json",
        gesture_models_dir=tmp_path,  # 앰비언트 체크포인트에 좌우되지 않게 빈 디렉토리
    )
    try:
        tabs = window.centralWidget()
        assert tabs is not None
        # six tabs: 실시간 / Gaze 파이프라인 / 손 추적 / 파이프라인 / 지연·어댑터 / 파인튜닝
        assert tabs.count() == 6
        assert tabs.tabText(0) == "실시간"
        assert tabs.tabText(1) == "Gaze 파이프라인"
        assert tabs.tabText(2) == "손 추적"
        assert tabs.tabText(3) == "파이프라인"
        assert tabs.tabText(4) == "지연·어댑터"
        assert tabs.tabText(5) == "파인튜닝"
        assert window._sample_button.text() == "시선 샘플 저장 (0/10)"
        assert window._sample_list.count() == 0
        assert window._gaze_config.enable_3d_target_matching is False
        assert window._gaze_config.require_3d_target_registration is False
        assert window._active_calibration_model is None
        assert window._register_target_button.text() == "물체 등록"
    finally:
        window.close()
        app.processEvents()


def test_hand_panel_renders_wrist_vectors_and_tracking_loss() -> None:
    """손 추적 탭이 손목 이동 속도·가속도 벡터를 표시하고, 추적 손실도 예외 없이 처리한다."""
    from jarvis.monitoring.app import HandPanel
    from jarvis.monitoring.hand_probe import HandSnapshot

    app = QApplication.instance() or QApplication([])
    panel = HandPanel("probe live", "제스처 인식 비활성", smoothing=True)
    try:
        detected = HandSnapshot(
            timestamp_ms=0,
            frame_id=1,
            hand_detected=True,
            handedness="Right",
            handedness_score=0.9,
            detection_confidence=0.9,
            palm_scale=0.2,
            image_points=None,
            model_points=None,
            model_points_raw=None,
            landmark_count=21,
            inference_ms=5.0,
            smoothed=True,
            wrist_velocity=(2.0, -1.0),
            wrist_acceleration=(0.5, 0.1),
        )
        panel.update_snapshot(detected)
        # 속도는 화살표(VectorArrowView, 2026-07-19)로 그려진다 — 픽셀 내용까지는
        # overlay.render_vector 단위 테스트가 검증하고, 여기서는 배선만 확인한다.
        assert not panel._velocity_view._canvas.pixmap().isNull()
        assert "‖·‖" in panel._accel_view._magnitude.text()

        lost = HandSnapshot(
            timestamp_ms=33,
            frame_id=2,
            hand_detected=False,
            handedness="",
            handedness_score=0.0,
            detection_confidence=0.0,
            palm_scale=0.0,
            image_points=None,
            model_points=None,
            model_points_raw=None,
            landmark_count=0,
            inference_ms=5.0,
            smoothed=True,
            wrist_velocity=None,
            wrist_acceleration=None,
        )
        panel.update_snapshot(lost)  # None 벡터 → 예외 없이 "no signal" 캔버스로 처리
        assert not panel._velocity_view._canvas.pixmap().isNull()
        assert "히스토리 없음" in panel._accel_view._magnitude.text()
    finally:
        panel.deleteLater()
        app.processEvents()


def test_target_registration_uses_auto_id(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        env={},
        start_camera=False,
        profiles_path=tmp_path / "profiles.json",
        samples_path=tmp_path / "samples.json",
        gesture_models_dir=tmp_path,
    )
    try:
        assert window._next_target_id() == "target_001"
    finally:
        window.close()
        app.processEvents()


def test_registration_guidance_prompts_diagonal_and_depth_motion() -> None:
    labels = [label for _end_ms, label in _REGISTRATION_GUIDANCE_PHASES]

    assert len(labels) == 5
    assert any("LEFT-UP" in label for label in labels)
    assert any("RIGHT-DOWN" in label for label in labels)
    assert any("LEFT-DOWN" in label for label in labels)
    assert any("RIGHT-UP" in label for label in labels)
    assert any("NEAR/FAR" in label for label in labels)


def test_startup_logs_gesture_recognition_off(tmp_path: Path) -> None:
    """학습 체크포인트가 없으면(빈 models_dir) 인식은 정직하게 off로 뜬다("미학습").

    gesture_models_dir을 빈 임시 디렉토리로 주입해 앰비언트 models/의 체크포인트 유무에
    좌우되지 않게 한다(있으면 인식이 켜진다 — 아래 별도 테스트).
    """
    app = QApplication.instance() or QApplication([])
    window = MainWindow(env={}, start_camera=False, gesture_models_dir=tmp_path)
    try:
        texts = [m.text for m in window._log.recent()]
        assert any("미학습" in t for t in texts)
        assert window._recognizer is None  # 인식기 미활성
    finally:
        window.close()
        app.processEvents()


def test_startup_activates_recognition_with_trained_checkpoint(tmp_path: Path) -> None:
    """학습된(trained=True) 체크포인트가 있으면 실시간 인식이 활성화된다(관측값 재사용 배선)."""
    import json

    torch = pytest.importorskip("torch")
    from jarvis.gesture_fusion.config import DEFAULT_GESTURE_CONFIG
    from jarvis.gesture_fusion.features import feature_dimension
    from jarvis.gesture_fusion.model import CausalTCN, ModelConfig

    net = CausalTCN(ModelConfig(feature_dim=feature_dimension(DEFAULT_GESTURE_CONFIG)))
    torch.save(net.state_dict(), tmp_path / "gesture_tcn_jester.pt")
    (tmp_path / "gesture_tcn_jester.pt.metadata.json").write_text(
        json.dumps({"version": "pretrain-epoch24", "trained": True}), encoding="utf-8"
    )

    app = QApplication.instance() or QApplication([])
    window = MainWindow(env={}, start_camera=False, gesture_models_dir=tmp_path)
    try:
        assert window._recognizer is not None
        assert window._gesture_source.available is True
        texts = [m.text for m in window._log.recent()]
        assert any("인식 활성" in t for t in texts)
    finally:
        window.close()
        app.processEvents()
