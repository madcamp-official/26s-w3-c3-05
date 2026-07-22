"""시연 패널 — 가독성(배경/글자색) 계약과 상태 갱신 배선을 검증한다.

이 앱에는 전역 다크 테마가 없다. 어두운 외관은 위젯마다 stylesheet로 만들기 때문에,
`background`만 지정하고 `color`를 빠뜨리면 글자가 시스템 기본색으로 떨어져 **검은
배경에 검은 글씨**가 된다(2026-07-22 실기기에서 '제스처'·'Intent' 두 줄이 그렇게
보이지 않았다 — 색을 따로 붙이지 않는 유일한 두 라벨이었다). 눈으로만 잡을 수 있는
버그라 여기서 규칙 자체를 고정한다.
"""

from __future__ import annotations

import os
import re
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from jarvis.monitoring.demo_bridge import PRESET_STRICT  # noqa: E402
from jarvis.monitoring.demo_panel import DemoPanel, TargetChoice  # noqa: E402
from jarvis.monitoring.virtual_bulb import VirtualBulbState  # noqa: E402


def _panel() -> DemoPanel:
    return DemoPanel(
        on_mapping_changed=lambda target_id, device_id: None,
        on_fallback_changed=lambda device_id: None,
        on_preset_changed=lambda preset: None,
        on_execution_toggled=lambda enabled: None,
    )


def _styles_with_background(root: QWidget) -> list[tuple[str, str]]:
    """`background`를 지정하는 모든 하위 위젯의 (클래스명, stylesheet)."""
    found: list[tuple[str, str]] = []
    for widget in [root, *root.findChildren(QWidget)]:
        sheet = widget.styleSheet()
        if sheet and re.search(r"\bbackground\s*:", sheet):
            found.append((type(widget).__name__, sheet))
    return found


def test_every_styled_background_also_sets_a_text_color() -> None:
    """배경을 칠하는 위젯은 글자색도 반드시 함께 지정한다."""
    app = QApplication.instance() or QApplication([])
    panel = _panel()
    try:
        offenders = [
            (name, sheet)
            for name, sheet in _styles_with_background(panel)
            if not re.search(r"\bcolor\s*:", sheet)
        ]
        assert not offenders, f"배경만 지정하고 글자색이 없는 위젯: {offenders}"
    finally:
        panel.deleteLater()
        app.processEvents()


def test_status_labels_keep_a_readable_color_after_updates() -> None:
    """상태가 갱신돼도 세 줄 모두 색이 남아 있어야 한다(색 없는 라벨이 생기지 않게)."""
    app = QApplication.instance() or QApplication([])
    panel = _panel()
    try:
        panel.set_state(
            locked="room.bulb",
            candidate=None,
            phase="TARGET_LOCKED",
            gesture="none",
            suppressed=True,
            raw_target="room.bulb",
        )
        for label in (
            panel._raw_target_label,
            panel._target_label,
            panel._gesture_label,
            panel._phase_label,
        ):
            assert re.search(r"\bcolor\s*:", label.styleSheet()), label.text()
    finally:
        panel.deleteLater()
        app.processEvents()


def test_state_text_reflects_lock_candidate_and_suppression() -> None:
    app = QApplication.instance() or QApplication([])
    panel = _panel()
    try:
        panel.set_state(locked=None, candidate=None, phase="IDLE", gesture="-", suppressed=False)
        assert "없음" in panel._target_label.text()

        panel.set_state(
            locked=None,
            candidate="laptop",
            phase="TARGET_CANDIDATE",
            gesture="-",
            suppressed=False,
        )
        assert "후보" in panel._target_label.text()

        panel.set_state(
            locked="room.bulb",
            candidate=None,
            phase="TARGET_LOCKED",
            gesture="stop_sign",
            suppressed=True,
        )
        assert "LOCKED" in panel._target_label.text()
        assert "커서 제어 정지" in panel._target_label.text()
        assert "stop_sign" in panel._gesture_label.text()
        assert "TARGET_LOCKED" in panel._phase_label.text()
    finally:
        panel.deleteLater()
        app.processEvents()


def test_raw_target_is_shown_separately_from_locked_device() -> None:
    """실시간 시선(원시 classifier 결과)과 바라보는 기기(Fusion dwell 확정)는
    서로 다른 값을 동시에 보여줄 수 있어야 한다 — 예: 아직 확정 전이라 '바라보는
    기기: 없음'이어도 지금 막 laptop을 보고 있다는 건 실시간 시선에 남는다."""
    app = QApplication.instance() or QApplication([])
    panel = _panel()
    try:
        panel.set_state(
            locked=None,
            candidate=None,
            phase="IDLE",
            gesture="-",
            suppressed=False,
            raw_target=None,
        )
        assert "없음" in panel._raw_target_label.text()
        assert "없음" in panel._target_label.text()

        panel.set_state(
            locked=None,
            candidate=None,
            phase="IDLE",
            gesture="-",
            suppressed=False,
            raw_target="laptop",
        )
        assert "laptop" in panel._raw_target_label.text()
        assert "없음" in panel._target_label.text()  # 아직 dwell 확정 전

        panel.set_state(
            locked="room.bulb",
            candidate=None,
            phase="TARGET_LOCKED",
            gesture="-",
            suppressed=False,
            raw_target="laptop",
        )
        # 확정된 기기(bulb)와 지금 실제로 보는 기기(laptop)가 달라도 둘 다
        # 각자의 값을 그대로 보여준다 — 서로를 덮어쓰지 않는다.
        assert "laptop" in panel._raw_target_label.text()
        assert "room.bulb" in panel._target_label.text()

        # raw_target을 생략하면(기존 호출부 호환) 조용히 '없음'으로 떨어진다.
        panel.set_state(locked=None, candidate=None, phase="IDLE", gesture="-", suppressed=False)
        assert "없음" in panel._raw_target_label.text()
    finally:
        panel.deleteLater()
        app.processEvents()


def test_mapping_table_rebuilds_without_leaking_rows() -> None:
    app = QApplication.instance() or QApplication([])
    panel = _panel()
    try:
        panel.set_targets([], {})
        panel.set_targets(
            [
                TargetChoice("target_001", "전구", "electric bulb"),
                TargetChoice("target_002", "노트북", "computer"),
            ],
            {"target_001": "room.bulb"},
        )
        assert panel._mapping_combos["target_001"].currentData() == "room.bulb"
        assert panel._mapping_combos["target_002"].currentText() == "(연결 안 함)"

        panel.set_targets([TargetChoice("target_003", "새 물체")], {})
        assert set(panel._mapping_combos) == {"target_003"}
    finally:
        panel.deleteLater()
        app.processEvents()


def test_registration_area_is_compact_and_hand_status_updates_live() -> None:
    """긴 등록 설명은 없애고 웹캠에서 옮긴 손 상태를 같은 자리에 보여준다."""
    app = QApplication.instance() or QApplication([])
    opened: list[bool] = []
    panel = DemoPanel(
        on_mapping_changed=lambda target_id, device_id: None,
        on_fallback_changed=lambda device_id: None,
        on_preset_changed=lambda preset: None,
        on_execution_toggled=lambda enabled: None,
        on_open_registration=lambda: opened.append(True),
    )
    try:
        assert panel._registration_button.text() == "등록 관리"
        panel._registration_button.click()
        assert opened == [True]

        pose = SimpleNamespace(label="none", confidence=0.97, trusted=True, reason="")
        snapshot = SimpleNamespace(
            hand_detected=True,
            handedness="Right",
            detection_confidence=0.99,
            palm_scale=0.265,
            smoothed=True,
            palm_tilt_degrees=4.2,
            pose=pose,
            pose_events=(),
            pose_state="",
        )
        panel.set_hand_status(snapshot, execution_enabled=True)
        status = panel._hand_status.text()
        assert "HAND  Right" in status
        assert "palm scale  0.265" in status
        assert "tilt 4°" in status
        assert "none 97%" in status
        assert "TCN 판정 대기 · 실행 ON" in status
    finally:
        panel.deleteLater()
        app.processEvents()


def test_execution_toggle_reports_armed_and_judgment_only_modes() -> None:
    app = QApplication.instance() or QApplication([])
    toggled: list[bool] = []
    panel = DemoPanel(
        on_mapping_changed=lambda target_id, device_id: None,
        on_fallback_changed=lambda device_id: None,
        on_preset_changed=lambda preset: None,
        on_execution_toggled=toggled.append,
    )
    try:
        # 기본 켜짐(사용자 지시, 2026-07-22) — 생성 중에는 connect 전이라 신호가
        # 안 뜨지만 상태 라벨은 이미 "실행 활성"으로 맞춰져 있어야 한다.
        assert panel._execution_toggle.isChecked() is True
        assert "실행 활성" in panel._execution_status.text()

        panel._execution_toggle.setChecked(False)
        assert toggled[-1] is False
        assert "판정 전용" in panel._execution_status.text()
        panel._execution_toggle.setChecked(True)
        assert toggled[-1] is True
        assert "실행 활성" in panel._execution_status.text()
    finally:
        panel.deleteLater()
        app.processEvents()


def test_fallback_combo_emits_runtime_id_not_display_label() -> None:
    app = QApplication.instance() or QApplication([])
    selected: list[str | None] = []
    panel = DemoPanel(
        on_mapping_changed=lambda target_id, device_id: None,
        on_fallback_changed=selected.append,
        on_preset_changed=lambda preset: None,
        on_execution_toggled=lambda enabled: None,
    )
    try:
        panel._fallback_combo.setCurrentIndex(1)
        panel._fallback_toggle.setChecked(True)
        assert selected[-1] == "room.bulb"
        assert "electric bulb" in panel._fallback_combo.currentText()
    finally:
        panel.deleteLater()
        app.processEvents()


def test_bulb_view_survives_every_state() -> None:
    """전원 꺼짐·경계값에서도 색 계산이 예외를 내지 않는다."""
    app = QApplication.instance() or QApplication([])
    panel = _panel()
    try:
        assert panel.bulb_view.width() == 56
        assert panel.bulb_view.height() == 56
        for state in (
            VirtualBulbState(power=False),
            VirtualBulbState(power=True, brightness=10, color_temperature=2700),
            VirtualBulbState(power=True, brightness=100, color_temperature=6500),
        ):
            panel.set_bulb(state, badge="미설정", ok=False)
            assert panel.bulb_view._bulb_color().isValid()
    finally:
        panel.deleteLater()
        app.processEvents()


def test_preset_callback_fires_with_selected_preset() -> None:
    app = QApplication.instance() or QApplication([])
    chosen: list[object] = []
    panel = DemoPanel(
        on_mapping_changed=lambda target_id, device_id: None,
        on_fallback_changed=lambda device_id: None,
        on_preset_changed=chosen.append,
        on_execution_toggled=lambda enabled: None,
    )
    try:
        panel._preset_combo.setCurrentIndex(2)  # 빡빡
        assert chosen and chosen[-1] is PRESET_STRICT
    finally:
        panel.deleteLater()
        app.processEvents()


def test_bulb_view_follows_the_active_color_mode() -> None:
    """색상 모드의 그림 색은 adapter가 기기로 보내는 RGB와 같아야 한다.

    화면과 실물이 서로 다른 색을 내면 시연에서 바로 들통난다 — 두 곳이 같은 변환
    (`wiz.hue_to_rgb`)을 쓰는지 고정한다.
    """
    from jarvis.runtime_protocol.adapters.wiz import hue_to_rgb

    app = QApplication.instance() or QApplication([])
    panel = _panel()
    try:
        panel.set_bulb(
            VirtualBulbState(power=True, brightness=100, color_mode=True, hue=120),
            badge="OK",
            ok=True,
        )
        assert panel.bulb_view._tint() == hue_to_rgb(120)

        # 색온도 모드로 돌아가면 색조 계산도 그쪽을 따른다.
        panel.set_bulb(
            VirtualBulbState(power=True, brightness=100, color_mode=False),
            badge="OK",
            ok=True,
        )
        assert panel.bulb_view._tint() != hue_to_rgb(120)
    finally:
        panel.deleteLater()
        app.processEvents()
