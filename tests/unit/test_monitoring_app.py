"""Offscreen smoke test: the desktop window constructs and wires its panels.

Runs Qt under the 'offscreen' platform so it needs no display. This does not
verify visual appearance — only that the widget tree builds, the tabs exist, and
teardown is clean. Requires the ui extra (PySide6).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from jarvis.monitoring.app import (  # noqa: E402
    MainWindow,
    _BOUNDARY_GUIDANCE_PHASES,
    _CENTER_GUIDANCE_PHASES,
)


def test_main_window_builds_offscreen(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        env={},
        start_camera=False,
        samples_path=tmp_path / "samples.json",
        gesture_models_dir=tmp_path,  # 앰비언트 체크포인트에 좌우되지 않게 빈 디렉토리
        diagnostics_dir=tmp_path / "diagnostics",
    )
    try:
        tabs = window.centralWidget()
        assert tabs is not None
        # eight tabs: 실시간 / 시선 인식 / Gaze 파이프라인 / 손 추적 / 파이프라인 /
        # 지연·어댑터 / 파인튜닝 / 시연
        assert tabs.count() == 8
        assert tabs.tabText(0) == "실시간"
        assert tabs.tabText(1) == "시선 인식"
        assert tabs.tabText(2) == "Gaze 파이프라인"
        assert tabs.tabText(3) == "손 추적"
        assert tabs.tabText(4) == "파이프라인"
        assert tabs.tabText(5) == "지연·어댑터"
        assert tabs.tabText(6) == "파인튜닝"
        assert tabs.tabText(7) == "시연"
        assert window._sample_button.text() == "시선 샘플 저장 (0/10)"
        assert window._sample_list.count() == 0
        assert window._gaze_config.enable_3d_target_matching is False
        assert window._gaze_config.require_3d_target_registration is False
        assert window._register_target_button.text() == "물체 등록"
        assert window._cancel_registration_button.isEnabled() is False
        assert window._registration_progress.value() == 0
        assert window._session_report_view.isReadOnly()
        assert window._session_report_button.text() == "최근 세션 분석"
        assert window._session_report_button.isEnabled() is False
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


def test_latest_session_report_is_rendered_in_window(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    path = diagnostics / "session_20260722_120000.jsonl"
    rows = [
        {
            "type": "header",
            "version": 2,
            "config": {"target_match_tolerance": 1.1},
            "targets": [],
        },
        {
            "type": "frame",
            "t": 100,
            "frame": 1,
            "label": "none",
            "obs": {
                "head": [0.0, 0.0, 0.0],
                "eyes_open": True,
                "face_scale": 0.1,
                "face_center": [0.5, 0.5],
            },
            "gaze": {"source": "head+iris", "source_reason": None, "feature": [0.0, 0.0]},
            "cls": {"target": "UNKNOWN", "reject": "outside every target"},
            "lock": {"locked": None, "dwell_ms": 0},
            "targets": {},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        env={},
        start_camera=False,
        samples_path=tmp_path / "samples.json",
        diagnostics_dir=diagnostics,
    )
    try:
        assert window._session_report_button.isEnabled()
        window._analyze_latest_session()
        rendered = window._session_report_view.toPlainText()
        assert "label: none" in rendered
        assert "accuracy(=UNKNOWN): 100%" in rendered
    finally:
        window.close()
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


def test_registration_guidance_covers_center_pose_and_boundary_edges() -> None:
    center_labels = [label for _end_ms, label, _video in _CENTER_GUIDANCE_PHASES]
    boundary_labels = [label for _end_ms, label, _video in _BOUNDARY_GUIDANCE_PHASES]

    # 1단계는 중앙 한 점 응시를 유지한 채 고개만 돌린다 — 테두리를 훑으며
    # 고개를 돌리면 pose 보정이 테두리 위치를 편향으로 오학습한다(gaze.md).
    assert len(center_labels) == 5
    assert all("응시" in label for label in center_labels)
    assert any("왼쪽" in label for label in center_labels)
    assert any("오른쪽" in label for label in center_labels)
    assert any("위·아래" in label for label in center_labels)
    assert any("가까이·멀리" in label for label in center_labels)
    assert not any("테두리" in label for label in center_labels)
    assert len(boundary_labels) == 5
    assert any("윗변" in label for label in boundary_labels)
    assert any("오른쪽 변" in label for label in boundary_labels)
    assert any("아랫변" in label for label in boundary_labels)
    assert any("왼쪽 변" in label for label in boundary_labels)


def test_registration_ui_starts_in_center_phase_and_can_cancel(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        env={},
        start_camera=False,
        profiles_path=tmp_path / "profiles.json",
        samples_path=tmp_path / "samples.json",
        gesture_models_dir=tmp_path,
    )
    try:
        window._begin_registration("target_001", "monitor", "UNKNOWN", "target_001")
        assert window._registration is not None
        assert "1/2" in window._registration_step.text()
        assert window._cancel_registration_button.isEnabled()
        assert not window._register_target_button.isEnabled()
        assert window._gaze_video._registration_guide is not None

        window._cancel_target_registration()
        assert window._registration is None
        assert window._cancel_registration_button.isEnabled() is False
        assert window._register_target_button.isEnabled()
        assert window._gaze_video._registration_guide is None
    finally:
        window.close()
        app.processEvents()


def test_startup_logs_gesture_recognition_off(tmp_path: Path) -> None:
    """The gesture pipeline exists but its model is untrained — startup says so
    honestly ("미학습"), never claiming recognition is running."""
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        env={},
        start_camera=False,
        gesture_models_dir=tmp_path,
    )
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


# --- 시연 탭: Gaze·Gesture → Fusion → 기기 명령 배선 ---------------------------


def _demo_window(tmp_path: Path):  # type: ignore[no-untyped-def]
    """시연 탭 검증용 창. 카메라·앰비언트 체크포인트에 좌우되지 않게 격리한다."""
    return MainWindow(
        env={},
        start_camera=False,
        samples_path=tmp_path / "samples.json",
        gesture_models_dir=tmp_path,
        diagnostics_dir=tmp_path / "diagnostics",
        profiles_path=tmp_path / "profiles.json",
    )


def _lost_hand_snapshot(frame_id: int):  # type: ignore[no-untyped-def]
    """pose_events가 있는 최소 HandSnapshot — 중재가 apply를 막는지만 본다."""
    from jarvis.gesture_fusion.pose_state import PoseEvent
    from jarvis.monitoring.hand_probe import HandSnapshot

    return HandSnapshot(
        timestamp_ms=frame_id * 33,
        frame_id=frame_id,
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
        wrist_velocity=None,
        wrist_acceleration=None,
        pose_events=(PoseEvent(kind="click", timestamp_ms=frame_id * 33),),
    )


def test_demo_tab_starts_with_execution_off(tmp_path: Path) -> None:
    """안전 기본값은 비실행 — 탭을 열었다는 이유로 기기가 움직이지 않는다."""
    app = QApplication.instance() or QApplication([])
    window = _demo_window(tmp_path)
    try:
        assert window._demo_panel.execution_enabled is False
        assert window._demo_bridge.execution_enabled is False
    finally:
        window.close()
        app.processEvents()


def test_demo_start_tab_selects_demo(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        env={},
        start_camera=False,
        samples_path=tmp_path / "samples.json",
        gesture_models_dir=tmp_path,
        profiles_path=tmp_path / "profiles.json",
        start_tab="시연",
    )
    try:
        tabs = window.centralWidget()
        assert tabs is not None
        assert tabs.tabText(tabs.currentIndex()) == "시연"
    finally:
        window.close()
        app.processEvents()


def test_pose_control_runs_when_not_locked_to_another_device(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = _demo_window(tmp_path)
    applied: list[int] = []
    window._pose_control.apply = lambda events: applied.append(len(events))  # type: ignore[method-assign]
    try:
        window._on_hand(_lost_hand_snapshot(1))
        assert applied == [1]
    finally:
        window.close()
        app.processEvents()


def test_pose_control_suppressed_while_locked_to_bulb(tmp_path: Path) -> None:
    """전구를 보는 동안 손을 움직여도 커서가 따라가면 안 된다(시선 lock 중재)."""
    from jarvis.contracts.messages import GestureEstimate, GesturePhase
    from jarvis.monitoring.demo_bridge import BULB_DEVICE_ID

    app = QApplication.instance() or QApplication([])
    window = _demo_window(tmp_path)
    applied: list[int] = []
    released: list[int] = []
    window._pose_control.apply = lambda events: applied.append(len(events))  # type: ignore[method-assign]
    window._pose_control.release = lambda: released.append(1)  # type: ignore[method-assign]
    try:
        window._demo_bridge.set_fallback(BULB_DEVICE_ID)
        for ms in range(0, 2000, 100):
            window._demo_bridge.push_gesture(
                GestureEstimate(
                    timestamp_ms=ms,
                    frame_id=ms // 10,
                    gesture="none",
                    gesture_confidence=0.1,
                    phase=GesturePhase.IDLE,
                    phase_confidence=0.5,
                    uncertainty=0.5,
                )
            )
        assert window._demo_bridge.locked_device == BULB_DEVICE_ID

        window._on_hand(_lost_hand_snapshot(1))
        assert applied == []  # 실행되지 않았다
        assert released  # 드래그가 눌린 채 남지 않게 놓았다
    finally:
        window.close()
        app.processEvents()


def test_demo_mapping_change_persists_and_resets_lock(tmp_path: Path) -> None:
    from jarvis.monitoring.demo_bridge import BULB_DEVICE_ID, DeviceMappingStore

    app = QApplication.instance() or QApplication([])
    window = _demo_window(tmp_path)
    try:
        window._on_demo_mapping_changed("target_001", BULB_DEVICE_ID)
        stored = DeviceMappingStore(tmp_path / "demo_device_map.json")
        assert stored.get("target_001") == BULB_DEVICE_ID
        assert window._demo_bridge.resolve_target("target_001") == BULB_DEVICE_ID
    finally:
        window.close()
        app.processEvents()


def test_demo_bulb_badge_reports_unconfigured(tmp_path: Path) -> None:
    """env가 비어 있으면 실물 전구는 '미설정'으로 정직하게 표시된다."""
    app = QApplication.instance() or QApplication([])
    window = _demo_window(tmp_path)
    try:
        badge, ok = window._last_bulb_badge
        assert ok is False
        assert "미설정" in badge
    finally:
        window.close()
        app.processEvents()


def test_demo_outcome_updates_virtual_bulb_but_flags_real_failure(tmp_path: Path) -> None:
    """실물 dispatch가 실패해도 가상 전구는 '보낸 명령'을 반영하고, 배지는 실패를 말한다."""
    from jarvis.contracts.messages import Intent
    from jarvis.runtime.executor import ExecutionOutcome, ExecutionStage

    app = QApplication.instance() or QApplication([])
    window = _demo_window(tmp_path)
    try:
        before = window._virtual_bulb.brightness
        window._on_execution_outcome(
            ExecutionOutcome(
                stage=ExecutionStage.DISPATCHED,
                detail="bulb did not answer",
                executed=False,
                intent=Intent(
                    intent_id="i-1",
                    target="room.bulb",
                    gesture="slide_two_fingers_down",
                    capability="brightness",
                    operation="decrement",
                    value=10,
                    target_confidence=0.9,
                    gesture_confidence=0.9,
                    expires_in_ms=1000,
                ),
                command_id="c-1",
                dispatch=None,
                rejection=None,
            )
        )
        assert window._virtual_bulb.brightness == before - 10
        badge, ok = window._last_bulb_badge
        assert ok is False
        assert "실패" in badge
    finally:
        window.close()
        app.processEvents()


def test_pose_release_called_once_on_suppression_entry(tmp_path: Path) -> None:
    """억제 중 매 프레임 release()를 부르면 macOS sink의 restore_dock이 초당 30번 돈다."""
    from jarvis.contracts.messages import GestureEstimate, GesturePhase
    from jarvis.monitoring.demo_bridge import BULB_DEVICE_ID

    app = QApplication.instance() or QApplication([])
    window = _demo_window(tmp_path)
    released: list[int] = []
    window._pose_control.apply = lambda events: None  # type: ignore[method-assign]
    window._pose_control.release = lambda: released.append(1)  # type: ignore[method-assign]
    try:
        window._demo_bridge.set_fallback(BULB_DEVICE_ID)
        for ms in range(0, 2000, 100):
            window._demo_bridge.push_gesture(
                GestureEstimate(
                    timestamp_ms=ms,
                    frame_id=ms // 10,
                    gesture="none",
                    gesture_confidence=0.1,
                    phase=GesturePhase.IDLE,
                    phase_confidence=0.5,
                    uncertainty=0.5,
                )
            )
        for frame_id in range(1, 6):
            window._on_hand(_lost_hand_snapshot(frame_id))
        assert released == [1]  # 5프레임 억제 동안 딱 한 번
    finally:
        window.close()
        app.processEvents()
