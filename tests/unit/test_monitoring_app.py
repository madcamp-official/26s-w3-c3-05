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

from jarvis.monitoring.app import MainWindow  # noqa: E402


def test_main_window_builds_offscreen(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(env={}, start_camera=False, samples_path=tmp_path / "samples.json")
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
    finally:
        window.close()
        app.processEvents()


def test_startup_logs_gesture_recognition_off() -> None:
    """The gesture pipeline exists but its model is untrained — startup says so
    honestly ("미학습"), never claiming recognition is running."""
    app = QApplication.instance() or QApplication([])
    window = MainWindow(env={}, start_camera=False)
    try:
        texts = [m.text for m in window._log.recent()]
        assert any("미학습" in t for t in texts)
    finally:
        window.close()
        app.processEvents()
