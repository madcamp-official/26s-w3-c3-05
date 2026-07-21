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

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
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


def test_space_toggles_recording_only_on_finetune_tab(tmp_path: Path) -> None:
    """스페이스바 녹화 토글은 파인튜닝 탭이 보일 때만 동작하고, 다른 탭에서는 무시된다.

    person_id 입력칸 등 다른 위젯이 포커스를 쥔 스페이스 입력과 충돌하면 안 되므로,
    실제 토글 로직(_toggle_recording)이 아니라 "언제 호출되는가"만 검증한다.
    """
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        env={},
        start_camera=False,
        samples_path=tmp_path / "samples.json",
        gesture_models_dir=tmp_path,
    )
    calls = 0

    def _fake_toggle() -> None:
        nonlocal calls
        calls += 1

    window._toggle_recording = _fake_toggle  # type: ignore[method-assign]
    try:
        window._tabs.setCurrentIndex(0)  # 실시간 탭 — 스페이스는 무시돼야 한다
        window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier))
        assert calls == 0

        window._tabs.setCurrentWidget(window._finetune_panel)
        window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier))
        assert calls == 1
    finally:
        window.close()
        app.processEvents()


def test_space_always_toggles_recording_even_when_combo_has_focus(tmp_path: Path) -> None:
    """스페이스바는 파인튜닝 탭에서 무조건 녹화 토글이다 — 동작 콤보에 포커스가 있어도

    콤보 팝업을 여는 대신 녹화만 토글된다(선택값은 그대로 유지). 2026-07-21 발견:
    QComboBox가 포커스를 쥔 채 스페이스를 누르면 Qt가 팝업을 직접 열면서 동시에
    이벤트가 MainWindow.keyPressEvent까지 전달돼 "콤보 열기 + 녹화 토글" 두 동작이
    겹쳤다 — `FinetuneRecordingPanel`의 이벤트 필터가 콤보에 이벤트가 전달되기 전에
    가로채 녹화 토글만 실행하고 소비하도록 고쳤다.
    """
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        env={},
        start_camera=False,
        samples_path=tmp_path / "samples.json",
        gesture_models_dir=tmp_path,
    )
    calls = 0

    def _fake_toggle() -> None:
        nonlocal calls
        calls += 1

    # 두 경로가 서로 다른 콜백 참조를 부른다: 이벤트 필터는 패널이 들고 있는
    # `_on_record_toggled`를, 창 레벨 keyPressEvent는 `window._toggle_recording`을
    # 직접 부른다 — 둘 다 패치해야 한 카운터로 양쪽 경로를 같이 검증할 수 있다.
    window._toggle_recording = _fake_toggle  # type: ignore[method-assign]
    window._finetune_panel._on_record_toggled = _fake_toggle  # type: ignore[method-assign]
    try:
        window._tabs.setCurrentWidget(window._finetune_panel)
        combo = window._finetune_panel._gesture_combo
        combo.setFocus()
        before = combo.currentText()

        app.sendEvent(
            combo, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        )

        assert calls == 1
        assert combo.currentText() == before  # 팝업이 안 열렸으므로 선택값 불변

        # 포커스가 콤보/버튼 밖(예: 아무 포커스도 없음)이면 MainWindow.keyPressEvent
        # 경로로도 여전히 토글돼야 한다.
        combo.clearFocus()
        window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier))
        assert calls == 2
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
