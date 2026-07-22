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

from PySide6.QtWidgets import QApplication, QInputDialog  # noqa: E402

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
        diagnostics_dir=tmp_path / "diagnostics",
    )
    try:
        tabs = window.centralWidget()
        assert tabs is not None
        # five tabs: 실시간 / Gaze 파이프라인 / 손 추적 / 파이프라인 / 지연·어댑터
        assert tabs.count() == 5
        assert tabs.tabText(0) == "실시간"
        assert tabs.tabText(1) == "Gaze 파이프라인"
        assert tabs.tabText(2) == "손 추적"
        assert tabs.tabText(3) == "파이프라인"
        assert tabs.tabText(4) == "지연·어댑터"
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
    try:
        window._start_target_registration()
        assert window._registration is not None
        assert window._registration.name == "테스트 물체"
        assert window._registration.device_type == device_type
        assert device_type in window._registration_status.text()
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
    )
    try:
        window._begin_registration("target_001", "monitor", "UNKNOWN", "target_001")
        assert window._registration is not None
        assert "1/2" in window._registration_step.text()
        assert window._cancel_registration_button.isEnabled()
        assert not window._register_target_button.isEnabled()
        assert window._video._registration_guide is not None

        window._cancel_target_registration()
        assert window._registration is None
        assert window._cancel_registration_button.isEnabled() is False
        assert window._register_target_button.isEnabled()
        assert window._video._registration_guide is None
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
    )
    try:
        texts = [m.text for m in window._log.recent()]
        assert any("미학습" in t for t in texts)
    finally:
        window.close()
        app.processEvents()
