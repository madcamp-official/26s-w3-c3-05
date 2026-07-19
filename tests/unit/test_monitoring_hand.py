"""Tests for the hand probe's honest degradation and status (no camera/model).

The probe does hand *tracking* only; it must never imply gesture *recognition*
is happening, and it must degrade honestly when the model/mediapipe is absent.
"""

from __future__ import annotations

from jarvis.monitoring.gesture_source import UntrainedGestureSource
from jarvis.monitoring.hand_probe import HandProbe


def test_probe_without_model_is_unavailable() -> None:
    probe = HandProbe(model_path=None)
    assert probe.start() is False
    assert probe.available is False
    assert "모델" in probe.status_text
    assert probe.process_bgr(object(), 0, 0) is None  # type: ignore[arg-type]


def test_gesture_recognition_status_is_honest_about_untrained_model() -> None:
    probe = HandProbe(model_path=None)
    status = probe.gesture_recognition_status
    assert "미학습" in status
    assert "비활성" in status
    # never claims recognition is active
    assert "인식됨" not in status


def test_untrained_gesture_source_yields_nothing() -> None:
    source = UntrainedGestureSource()
    assert source.available is False
    assert source.poll() == []
    assert "미학습" in source.status_text


def test_display_smoothing_defaults_on_and_toggles() -> None:
    probe = HandProbe(model_path=None)
    assert probe.smoothing is True  # smoothed vertices shown by default
    probe.set_smoothing(False)
    assert probe.smoothing is False
    probe.set_smoothing(True)
    assert probe.smoothing is True
