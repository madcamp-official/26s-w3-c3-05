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

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402

from jarvis.monitoring.app import (  # noqa: E402
    MainWindow,
    VideoView,
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
        window.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        )
        assert calls == 0

        window._tabs.setCurrentWidget(window._finetune_panel)
        window.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        )
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
        window.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        )
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


@pytest.mark.parametrize("device_type", ["computer", "electric bulb"])
def test_target_registration_selects_device_type_from_dropdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        env={},
        start_camera=False,
        profiles_path=tmp_path / "profiles.json",
        samples_path=tmp_path / "samples.json",
        gesture_models_dir=tmp_path,
    )
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        lambda *_args, **_kwargs: ("테스트 물체", True),
    )

    def choose_type(
        _parent: object,
        _title: str,
        _label: str,
        items: list[str],
        current: int,
        editable: bool,
    ) -> tuple[str, bool]:
        assert items == ["computer", "electric bulb"]
        assert current == 0
        assert editable is False
        return device_type, True

    monkeypatch.setattr(QInputDialog, "getItem", choose_type)

    # computer는 끄덕임 게이트 여부를 실제 모달(QMessageBox.question)로 묻는다
    # (2026-07-22) — mock하지 않으면 자동 테스트가 응답 없는 모달에서 멈춘다.
    # electric bulb는 이 다이얼로그를 아예 띄우지 않으므로 호출되면 실패시켜
    # 그 가정이 깨지지 않는지도 함께 확인한다.
    def answer_nod_gate_prompt(*_args: object, **_kwargs: object) -> QMessageBox.StandardButton:
        assert device_type == "computer", "electric bulb는 끄덕임 게이트를 묻지 않아야 한다"
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", answer_nod_gate_prompt)
    try:
        window._start_target_registration()
        assert window._registration is not None
        assert window._registration.name == "테스트 물체"
        assert window._registration.device_type == device_type
        assert device_type in window._registration_status.text()
        assert window._registration.requires_nod_gate == (device_type == "computer")
    finally:
        window._cancel_target_registration()
        window.close()
        app.processEvents()


def test_target_registration_cancelled_when_device_type_is_not_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        env={},
        start_camera=False,
        profiles_path=tmp_path / "profiles.json",
        samples_path=tmp_path / "samples.json",
        gesture_models_dir=tmp_path,
    )
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        lambda *_args, **_kwargs: ("테스트 물체", True),
    )
    monkeypatch.setattr(
        QInputDialog,
        "getItem",
        lambda *_args, **_kwargs: ("", False),
    )
    try:
        window._start_target_registration()
        assert window._registration is None
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


def test_demo_tab_starts_ready_for_laptop_control(tmp_path: Path) -> None:
    """시연 기본값(2026-07-22 사용자 지시): 실행 켜짐 + 타깃 고정 laptop.

    동적 제스처로 노트북을 바로 제어할 수 있도록, 탭을 열면 실행 스위치가 켜져
    있고 타깃이 laptop에 고정돼 있다. 패널 체크박스와 브릿지 상태가 일치해야
    한다(패널 초기 상태를 단일 소스로 브릿지에 동기화한 결과).
    """
    from jarvis.monitoring.demo_bridge import LAPTOP_DEVICE_ID

    app = QApplication.instance() or QApplication([])
    window = _demo_window(tmp_path)
    try:
        assert window._demo_panel.execution_enabled is True
        assert window._demo_bridge.execution_enabled is True
        assert window._demo_panel.fallback_device == LAPTOP_DEVICE_ID
        assert window._demo_bridge.fallback_device == LAPTOP_DEVICE_ID
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


def test_demo_raw_target_reflects_latest_gaze_and_fallback(tmp_path: Path) -> None:
    """실시간 시선(원시 classifier 결과)은 push_target()이 실제로 쓰는 값과
    항상 일치해야 한다 — 특히 타깃 고정(fallback)이 켜져 있으면 push_target()은
    합성 estimate로 바뀌므로 화면도 원시 시선 대신 고정 기기를 보여준다.
    시연 탭은 기본적으로 laptop 고정이 켜진 채 시작한다(2026-07-22)."""
    from types import SimpleNamespace

    from jarvis.monitoring.demo_bridge import BULB_DEVICE_ID, LAPTOP_DEVICE_ID, UNKNOWN_TARGET

    app = QApplication.instance() or QApplication([])
    window = _demo_window(tmp_path)
    try:
        window._on_demo_mapping_changed("target_001", BULB_DEVICE_ID)

        # 기본값(laptop 고정)이 켜져 있는 동안은 원시 시선과 무관하게 고정
        # 기기를 보여준다 — push_target()이 실제로 쓰는 합성값과 일치해야 한다.
        window._latest_gaze = SimpleNamespace(target_estimate=SimpleNamespace(target="target_001"))
        window._update_demo_state("-")
        assert window._demo_panel._raw_target_label.text() == f"실시간 시선: {LAPTOP_DEVICE_ID}"

        # 고정을 끄면 원시 gaze 결과가 그대로 드러난다.
        window._demo_bridge.set_fallback(None)
        window._update_demo_state("-")
        assert window._demo_panel._raw_target_label.text() == f"실시간 시선: {BULB_DEVICE_ID}"

        window._latest_gaze = SimpleNamespace(
            target_estimate=SimpleNamespace(target=UNKNOWN_TARGET)
        )
        window._update_demo_state("-")
        assert "없음" in window._demo_panel._raw_target_label.text()

        window._demo_bridge.set_fallback(BULB_DEVICE_ID)
        window._latest_gaze = SimpleNamespace(target_estimate=SimpleNamespace(target="target_001"))
        window._update_demo_state("-")
        assert window._demo_panel._raw_target_label.text() == f"실시간 시선: {BULB_DEVICE_ID}"
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


def test_demo_probe_state_replaces_the_screen_value(tmp_path: Path) -> None:
    """실물에서 읽은 상태가 화면의 단일 소스가 된다 — 예전에는 이 값을 버려서 화면이
    임의의 초기값(60%·4000K)에 머물렀고 실물과 하나도 맞지 않았다."""
    from jarvis.monitoring.bulb_probe import BulbProbeResult
    from jarvis.monitoring.virtual_bulb import state_from_pilot

    app = QApplication.instance() or QApplication([])
    window = _demo_window(tmp_path)
    try:
        assert window._bulb_verified is False
        state = state_from_pilot({"state": True, "dimming": 30, "r": 255, "g": 0, "b": 0})
        window._on_bulb_probed(BulbProbeResult(True, "전구 연결됨", state))
        assert window._bulb_verified is True
        assert window._virtual_bulb.brightness == 30
        assert window._virtual_bulb.color_mode is True
    finally:
        window.close()
        app.processEvents()


def test_demo_unreadable_probe_does_not_claim_the_screen_is_real(tmp_path: Path) -> None:
    """상태를 못 읽었으면 '실물에서 읽은 값'이라고 주장하지 않는다."""
    from jarvis.monitoring.bulb_probe import BulbProbeResult

    app = QApplication.instance() or QApplication([])
    window = _demo_window(tmp_path)
    try:
        window._bulb_verified = True
        window._on_bulb_probed(BulbProbeResult(False, "전구 연결 실패", None))
        assert window._bulb_verified is False
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


def test_video_view_can_show_only_hand_tracking_without_other_debug_overlays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """시연 탭은 HUD·gaze를 숨기면서 MediaPipe 손 입력만 선택적으로 남긴다."""
    import jarvis.monitoring.app as app_module

    calls: list[str] = []
    hand_detail_flags: list[bool] = []
    monkeypatch.setattr(app_module, "draw_hud", lambda *a, **k: calls.append("hud"))
    monkeypatch.setattr(app_module, "draw_gaze_overlay", lambda *a, **k: calls.append("gaze"))

    def _record_hand(*args: object, **kwargs: object) -> None:
        calls.append("hand")
        hand_detail_flags.append(bool(kwargs["show_details"]))

    monkeypatch.setattr(app_module, "draw_hand_overlay", _record_hand)
    monkeypatch.setattr(
        app_module,
        "draw_registration_guidance",
        lambda *a, **k: calls.append("registration"),
    )

    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    clean = VideoView(show_overlay=False)
    clean.set_gaze(object())
    clean.set_hand(object())
    clean.set_registration_guidance("t", "i", 0.5)
    clean.show_frame(frame)
    assert calls == []

    hand_only = VideoView(show_overlay=False, show_hand_overlay=True)
    hand_only.set_gaze(object())
    hand_only.set_hand(object())
    hand_only.set_registration_guidance("t", "i", 0.5)
    hand_only.show_frame(frame)
    assert calls == ["hand"]
    assert hand_detail_flags[-1] is True
    calls.clear()

    skeleton_only = VideoView(
        show_overlay=False,
        show_hand_overlay=True,
        show_hand_details=False,
    )
    skeleton_only.set_hand(object())
    skeleton_only.show_frame(frame)
    assert calls == ["hand"]
    assert hand_detail_flags[-1] is False
    calls.clear()

    debug = VideoView()  # 기본값 True — 다른 탭은 여전히 다 그려야 한다.
    debug.set_gaze(object())
    debug.set_hand(object())
    debug.set_registration_guidance("t", "i", 0.5)
    debug.show_frame(frame)
    assert calls == ["hud", "gaze", "registration", "hand"]


def test_demo_tab_video_keeps_only_hand_overlay(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        env={},
        start_camera=False,
        profiles_path=tmp_path / "profiles.json",
        samples_path=tmp_path / "samples.json",
    )
    try:
        assert window._demo_video._show_overlay is False
        assert window._demo_video._show_hand_overlay is True
        assert window._demo_video._show_hand_details is False
        assert window._video._show_overlay is True
    finally:
        window.close()
        app.processEvents()


def test_demo_registration_button_opens_gaze_registration_tab(tmp_path: Path) -> None:
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
        window._demo_panel._registration_button.click()
        tabs = window.centralWidget()
        assert tabs is not None
        assert tabs.tabText(tabs.currentIndex()) == "시선 인식"
    finally:
        window.close()
        app.processEvents()


@pytest.mark.parametrize(
    ("device_type", "runtime_device"),
    [("computer", "laptop"), ("electric bulb", "room.bulb")],
)
def test_registered_device_type_is_automatically_mapped_for_demo(
    tmp_path: Path, device_type: str, runtime_device: str
) -> None:
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps(
            [
                {
                    "target_id": "target_001",
                    "name": "demo device",
                    "device_type": device_type,
                    "direction": {"yaw": 0.0, "pitch": 0.0},
                    "spread": {"yaw": 4.0, "pitch": 4.0},
                    "device_id": "target_001",
                }
            ]
        ),
        encoding="utf-8",
    )
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        env={},
        start_camera=False,
        samples_path=tmp_path / "samples.json",
        gesture_models_dir=tmp_path,
        profiles_path=profiles,
    )
    try:
        assert window._device_mapping.get("target_001") == runtime_device
        assert window._demo_panel._mapping_combos["target_001"].currentData() == runtime_device
    finally:
        window.close()
        app.processEvents()


def test_demo_execution_toggle_arms_and_disarms_command_dispatch(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = _demo_window(tmp_path)
    try:
        window._demo_panel._execution_toggle.setChecked(True)
        assert window._demo_bridge.execution_enabled is True
        window._demo_panel._execution_toggle.setChecked(False)
        assert window._demo_bridge.execution_enabled is False
    finally:
        window.close()
        app.processEvents()


def _committed_decision(target: str, gesture: str = "slide_two_fingers_left"):  # type: ignore[no-untyped-def]
    """실행 게이트만 보기 위한 최소 커밋 판정(점수는 게이트가 안 읽으므로 None)."""
    from jarvis.gesture_fusion.fusion import CommitDecision

    return CommitDecision(
        committed=True,
        reason="committed",
        target=target,
        gesture=gesture,
        score=None,
        timestamp_ms=1000,
        frame_id=1,
        intent_id="intent-1",
    )


def test_laptop_command_blocked_while_computer_control_is_off(tmp_path: Path) -> None:
    """"손동작으로 컴퓨터 제어"가 꺼져 있으면 노트북 대상 명령은 실행되지 않는다.

    2026-07-22 회귀: TCN→Fusion→Intent 경로가 정적 pose 제어와 배선이 달라 그 토글을
    통과하지 않았고, 체크를 꺼도 두 손가락 slide가 실제로 데스크톱을 전환했다.
    """
    app = QApplication.instance() or QApplication([])
    window = _demo_window(tmp_path)
    submitted: list[object] = []
    try:
        window._demo_bridge.execution_enabled = True
        window._execute_worker = type("W", (), {"submit": lambda _s, d: submitted.append(d)})()

        window._set_control_enabled(False)
        window._handle_commit_decision(_committed_decision("laptop"))
        assert submitted == []  # 컴퓨터 제어가 꺼져 있으므로 실행 금지

        window._set_control_enabled(True)
        window._handle_commit_decision(_committed_decision("laptop"))
        assert len(submitted) == 1  # 켜면 평소대로 실행
    finally:
        window._execute_worker = None
        window.close()
        app.processEvents()


def test_bulb_command_ignores_computer_control_toggle(tmp_path: Path) -> None:
    """전구는 OS 입력이 아니라 네트워크 명령이라 컴퓨터 제어 토글의 대상이 아니다."""
    app = QApplication.instance() or QApplication([])
    window = _demo_window(tmp_path)
    submitted: list[object] = []
    try:
        window._demo_bridge.execution_enabled = True
        window._execute_worker = type("W", (), {"submit": lambda _s, d: submitted.append(d)})()

        window._set_control_enabled(False)
        window._handle_commit_decision(_committed_decision("room.bulb", "rotate_clockwise"))
        assert len(submitted) == 1
    finally:
        window._execute_worker = None
        window.close()
        app.processEvents()


def test_demo_tab_runs_static_pose_control_for_laptop(tmp_path: Path) -> None:
    """노트북(컴퓨터) 제어를 전부 정적 포즈로 통일했으므로(2026-07-22, 사용자 지시),
    시연 탭이라도 노트북 맥락(전구 lock이 아님)이고 실행이 켜져 있으면 정적 pose 이벤트를
    실행한다. 노트북에는 동적 capability 매핑이 없어(laptop {}) 이중 실행 위험이 없다.

    전구를 보는 중(should_suppress_pose=True)에는 계속 억제한다 — 전구를 보며 손을 움직일
    때 커서가 따라가면 안 되기 때문이다."""
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        env={},
        start_camera=False,
        samples_path=tmp_path / "samples.json",
        gesture_models_dir=tmp_path,
        profiles_path=tmp_path / "profiles.json",
        start_tab="시연",
    )
    applied: list[int] = []
    window._pose_control.apply = lambda events: applied.append(len(events))  # type: ignore[method-assign]
    # should_suppress_pose(전구 lock 여부)를 통제해 두 경로를 모두 검증한다.
    bridge_cls = type(window._demo_bridge)
    original_suppress = bridge_cls.should_suppress_pose
    try:
        window._demo_panel._execution_toggle.setChecked(True)
        # 노트북 맥락(전구 lock 아님): 정적 pose 이벤트(click)가 실행 경로로 나간다.
        bridge_cls.should_suppress_pose = property(lambda self: False)  # type: ignore[assignment]
        window._on_hand(_lost_hand_snapshot(1))
        assert applied == [1]
        assert window._demo_bridge.execution_enabled is True

        # 전구 lock 중이면(should_suppress_pose=True) 억제한다 — 커서가 안 따라가게.
        applied.clear()
        bridge_cls.should_suppress_pose = property(lambda self: True)  # type: ignore[assignment]
        window._on_hand(_lost_hand_snapshot(2))
        assert applied == []
    finally:
        bridge_cls.should_suppress_pose = original_suppress  # type: ignore[assignment]
        window.close()
        app.processEvents()


def test_static_swipe_routes_to_bulb_brightness() -> None:
    """전구 lock 중 정적 좌우 스와이프가 전구 좌우 슬라이드(밝기)로 매핑된다(2026-07-22).

    desktop_prev(왼쪽)=밝기 감소, desktop_next(오른쪽)=밝기 증가. 합성 CommitDecision의
    gesture가 실제 capability map에서 room.bulb 밝기 명령으로 해석되는지(단일 진실원)까지
    확인한다 — GUI 창 없이 클래스 속성·정적 메서드만 검증한다."""
    from jarvis.runtime.devices import build_default_capability_map

    cap = build_default_capability_map()

    class _Snap:
        timestamp_ms = 100
        frame_id = 3

    for kind, operation in (("desktop_prev", "decrement"), ("desktop_next", "increment")):
        gesture = MainWindow._SWIPE_TO_SLIDE[kind]
        decision = MainWindow._synthetic_swipe_decision("room.bulb", gesture, _Snap())
        assert decision.committed is True
        assert decision.target == "room.bulb"
        assert decision.intent_id  # 고유 id가 있어야 실행기가 dedup으로 버리지 않는다
        action = cap.lookup("room.bulb", gesture)
        assert action is not None
        assert action.capability == "brightness"
        assert action.operation == operation
